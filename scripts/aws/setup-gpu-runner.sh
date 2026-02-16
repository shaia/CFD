#!/usr/bin/env bash
# setup-gpu-runner.sh — One-time AWS resource setup for EC2 GPU CI runner
#
# Creates:
#   - IAM policy + user (cfd-github-runner) with minimal EC2 permissions
#   - Security group (cfd-gpu-runner-sg) with SSH + outbound HTTPS
#   - EC2 key pair for SSH access
#
# Prerequisites:
#   - AWS CLI v2 configured (aws configure)
#   - Permissions to create IAM users, security groups, and key pairs
#
# Usage:
#   ./scripts/aws/setup-gpu-runner.sh [--region us-east-1] [--vpc-id vpc-xxx]
#
# After running, add the printed values as GitHub repository secrets:
#   Settings > Secrets and variables > Actions > New repository secret

set -euo pipefail

# Keep window open on error so the message is visible
trap 'echo ""; echo "ERROR: Script failed at line $LINENO. See above for details."; echo "Press Enter to close..."; read' ERR

# ---------- Configuration ----------
REGION="${AWS_DEFAULT_REGION:-us-east-1}"
PROJECT_NAME="cfd-gpu-runner"
IAM_USER="${PROJECT_NAME}"
IAM_POLICY="${PROJECT_NAME}-policy"
SG_NAME="${PROJECT_NAME}-sg"
KEY_NAME="${PROJECT_NAME}"
CONFIG_FILE="scripts/aws/.gpu-runner-config"

# ---------- Parse Arguments ----------
VPC_ID=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --region) REGION="$2"; shift 2 ;;
        --vpc-id) VPC_ID="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== CFD GPU Runner — AWS Setup ==="
echo "Region: ${REGION}"
echo ""

# ---------- Helpers ----------
check_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: '$1' not found. Please install it first."
        exit 1
    fi
}

check_command aws

# Verify AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    echo "ERROR: AWS CLI not configured. Run 'aws configure' first."
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account: ${ACCOUNT_ID}"
echo ""

# ---------- 1. Get VPC and Subnet ----------
echo "[1/4] Resolving VPC and subnet..."

if [[ -z "${VPC_ID}" ]]; then
    VPC_ID=$(aws ec2 describe-vpcs \
        --region "${REGION}" \
        --filters "Name=isDefault,Values=true" \
        --query "Vpcs[0].VpcId" \
        --output text 2>/dev/null || echo "None")

    if [[ "${VPC_ID}" == "None" || -z "${VPC_ID}" ]]; then
        echo "ERROR: No default VPC found in ${REGION}. Specify --vpc-id."
        exit 1
    fi
fi

SUBNET_ID=$(aws ec2 describe-subnets \
    --region "${REGION}" \
    --filters "Name=vpc-id,Values=${VPC_ID}" "Name=default-for-az,Values=true" \
    --query "Subnets[0].SubnetId" \
    --output text 2>/dev/null || echo "None")

if [[ "${SUBNET_ID}" == "None" || -z "${SUBNET_ID}" ]]; then
    # Fall back to any subnet in the VPC
    SUBNET_ID=$(aws ec2 describe-subnets \
        --region "${REGION}" \
        --filters "Name=vpc-id,Values=${VPC_ID}" \
        --query "Subnets[0].SubnetId" \
        --output text)
fi

echo "  VPC:    ${VPC_ID}"
echo "  Subnet: ${SUBNET_ID}"

# ---------- 2. Create Security Group ----------
echo ""
echo "[2/4] Creating security group..."

SG_ID=$(aws ec2 describe-security-groups \
    --region "${REGION}" \
    --filters "Name=group-name,Values=${SG_NAME}" "Name=vpc-id,Values=${VPC_ID}" \
    --query "SecurityGroups[0].GroupId" \
    --output text 2>/dev/null || echo "None")

if [[ "${SG_ID}" != "None" && -n "${SG_ID}" ]]; then
    echo "  Security group already exists: ${SG_ID}"
else
    SG_ID=$(aws ec2 create-security-group \
        --region "${REGION}" \
        --group-name "${SG_NAME}" \
        --description "CFD GPU runner - SSH + outbound HTTPS" \
        --vpc-id "${VPC_ID}" \
        --query "GroupId" \
        --output text)

    # SSH inbound from caller's IP
    MY_IP=$(curl -s https://checkip.amazonaws.com)
    aws ec2 authorize-security-group-ingress \
        --region "${REGION}" \
        --group-id "${SG_ID}" \
        --protocol tcp --port 22 \
        --cidr "${MY_IP}/32" \
        --no-cli-pager

    echo "  Created: ${SG_ID} (SSH from ${MY_IP}/32)"
fi

# ---------- 3. Create Key Pair ----------
echo ""
echo "[3/4] Creating EC2 key pair..."

KEY_FILE="${KEY_NAME}.pem"

if aws ec2 describe-key-pairs --region "${REGION}" --key-names "${KEY_NAME}" &>/dev/null; then
    echo "  Key pair '${KEY_NAME}' already exists"
    if [[ ! -f "${KEY_FILE}" ]]; then
        echo "  WARNING: ${KEY_FILE} not found locally. You may need to recreate the key pair."
    fi
else
    aws ec2 create-key-pair \
        --region "${REGION}" \
        --key-name "${KEY_NAME}" \
        --query "KeyMaterial" \
        --output text > "${KEY_FILE}"
    chmod 400 "${KEY_FILE}"
    echo "  Created: ${KEY_NAME} (saved to ${KEY_FILE})"
fi

# ---------- 4. Create IAM User + Policy ----------
echo ""
echo "[4/4] Creating IAM user and policy..."

# Create the policy document
POLICY_DOC=$(cat <<'POLICY'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "EC2RunnerManagement",
            "Effect": "Allow",
            "Action": [
                "ec2:RunInstances",
                "ec2:TerminateInstances",
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceStatus",
                "ec2:CreateTags",
                "ec2:DescribeSubnets",
                "ec2:DescribeSecurityGroups",
                "ec2:DescribeImages"
            ],
            "Resource": "*"
        }
    ]
}
POLICY
)

# Create or find the policy
POLICY_ARN=$(aws iam list-policies \
    --scope Local \
    --query "Policies[?PolicyName=='${IAM_POLICY}'].Arn | [0]" \
    --output text 2>/dev/null || echo "None")

if [[ "${POLICY_ARN}" == "None" || -z "${POLICY_ARN}" ]]; then
    POLICY_ARN=$(aws iam create-policy \
        --policy-name "${IAM_POLICY}" \
        --policy-document "${POLICY_DOC}" \
        --query "Policy.Arn" \
        --output text)
    echo "  Created policy: ${POLICY_ARN}"
else
    echo "  Policy already exists: ${POLICY_ARN}"
fi

# Create or find the user
if aws iam get-user --user-name "${IAM_USER}" &>/dev/null; then
    echo "  IAM user '${IAM_USER}' already exists"
else
    aws iam create-user --user-name "${IAM_USER}" --no-cli-pager
    echo "  Created user: ${IAM_USER}"
fi

# Attach policy
aws iam attach-user-policy \
    --user-name "${IAM_USER}" \
    --policy-arn "${POLICY_ARN}" 2>/dev/null || true

# Create access key (only if none exist)
EXISTING_KEYS=$(aws iam list-access-keys \
    --user-name "${IAM_USER}" \
    --query "AccessKeyMetadata | length(@)" \
    --output text)

if [[ "${EXISTING_KEYS}" == "0" ]]; then
    ACCESS_KEY_JSON=$(aws iam create-access-key \
        --user-name "${IAM_USER}" \
        --output json)
    ACCESS_KEY_ID=$(echo "${ACCESS_KEY_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['AccessKey']['AccessKeyId'])")
    SECRET_ACCESS_KEY=$(echo "${ACCESS_KEY_JSON}" | python3 -c "import sys,json; print(json.load(sys.stdin)['AccessKey']['SecretAccessKey'])")
    echo "  Created access key: ${ACCESS_KEY_ID}"
else
    echo "  Access key already exists (${EXISTING_KEYS} key(s)). Skipping creation."
    ACCESS_KEY_ID="<existing — check AWS console>"
    SECRET_ACCESS_KEY="<existing — check AWS console>"
fi

# ---------- Find NVIDIA Deep Learning AMI ----------
echo ""
echo "Looking up NVIDIA Deep Learning AMI..."

AMI_ID=$(aws ec2 describe-images \
    --region "${REGION}" \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning AMI GPU PyTorch*Ubuntu 22.04*" \
        "Name=state,Values=available" \
        "Name=architecture,Values=x86_64" \
    --query "Images | sort_by(@, &CreationDate) | [-1].ImageId" \
    --output text 2>/dev/null || echo "None")

if [[ "${AMI_ID}" == "None" || -z "${AMI_ID}" ]]; then
    # Fallback: NVIDIA GPU-Optimized AMI
    AMI_ID=$(aws ec2 describe-images \
        --region "${REGION}" \
        --owners amazon \
        --filters \
            "Name=name,Values=*NVIDIA*Ubuntu*22.04*" \
            "Name=state,Values=available" \
            "Name=architecture,Values=x86_64" \
        --query "Images | sort_by(@, &CreationDate) | [-1].ImageId" \
        --output text 2>/dev/null || echo "NOT_FOUND")
fi

echo "  AMI: ${AMI_ID}"

# ---------- Save Config ----------
mkdir -p "$(dirname "${CONFIG_FILE}")"
cat > "${CONFIG_FILE}" <<EOF
# CFD GPU Runner — AWS Configuration
# Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")
REGION=${REGION}
VPC_ID=${VPC_ID}
SUBNET_ID=${SUBNET_ID}
SG_ID=${SG_ID}
KEY_NAME=${KEY_NAME}
AMI_ID=${AMI_ID}
IAM_USER=${IAM_USER}
EOF
echo ""
echo "Config saved to: ${CONFIG_FILE}"

# ---------- Summary ----------
echo ""
echo "============================================"
echo "  AWS Setup Complete"
echo "============================================"
echo ""
echo "Add these GitHub repository secrets:"
echo "  Settings > Secrets and variables > Actions"
echo ""
echo "  AWS_ACCESS_KEY_ID       = ${ACCESS_KEY_ID}"
echo "  AWS_SECRET_ACCESS_KEY   = ${SECRET_ACCESS_KEY}"
echo "  EC2_SUBNET_ID           = ${SUBNET_ID}"
echo "  EC2_SECURITY_GROUP_ID   = ${SG_ID}"
echo "  EC2_AMI_ID              = ${AMI_ID}"
echo "  GH_PERSONAL_ACCESS_TOKEN = <create at github.com/settings/tokens with 'repo' scope>"
echo ""
echo "To manually launch an instance for debugging:"
echo ""
echo "  aws ec2 run-instances \\"
echo "    --region ${REGION} \\"
echo "    --image-id ${AMI_ID} \\"
echo "    --instance-type g4dn.4xlarge \\"
echo "    --key-name ${KEY_NAME} \\"
echo "    --security-group-ids ${SG_ID} \\"
echo "    --subnet-id ${SUBNET_ID} \\"
echo "    --associate-public-ip-address \\"
echo "    --block-device-mappings '[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":100}}]' \\"
echo "    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=cfd-gpu-dev}]'"
echo ""
echo "Then SSH in:"
echo "  ssh -i ${KEY_FILE} ubuntu@<public-ip>"
echo ""
echo "Press Enter to close..."
read
