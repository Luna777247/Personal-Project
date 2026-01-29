# Azure Cloud Security Lab

## Overview

This lab demonstrates common cloud security misconfigurations in Azure and provides remediation steps. The lab creates a simple cloud environment with intentional security gaps for educational purposes.

## Lab Components

### 1. Resource Group Creation
### 2. Storage Account with Misconfigurations
### 3. Virtual Machine with Security Issues
### 4. Access Control and Identity Problems
### 5. Logging and Monitoring Gaps

## Prerequisites

- Azure CLI installed and configured
- Azure subscription with sufficient permissions
- Python with azure-mgmt-* packages

## Lab Setup Script

```bash
# Login to Azure
az login

# Set subscription
az account set --subscription "your-subscription-id"

# Run the lab setup
python cloud_lab/setup_azure_lab.py
```

## Security Misconfigurations Demonstrated

### 1. Storage Account Public Access

**Issue:** Blob containers with public read access
**Risk:** Data exposure to unauthorized users
**Detection:**
```bash
az storage account list --query "[].{name:name, allowBlobPublicAccess:allowBlobPublicAccess}" -o table
```

**Remediation:**
```bash
az storage account update --name mystorageaccount --resource-group myrg --allow-blob-public-access false
```

### 2. VM with Open RDP/SSH Ports

**Issue:** Virtual machines with unrestricted access
**Risk:** Brute force attacks and unauthorized access
**Detection:**
```bash
az network nsg rule list --resource-group myrg --nsg-name mynsg --query "[].{name:name, access:access, direction:direction, destinationPortRange:destinationPortRange}" -o table
```

**Remediation:**
```bash
az network nsg rule update --resource-group myrg --nsg mynsg --name RDP --access Deny
```

### 3. Missing Azure Monitor Diagnostics

**Issue:** No logging enabled for resources
**Risk:** Unable to detect security incidents
**Detection:**
```bash
az monitor diagnostic-settings list --resource /subscriptions/.../resourceGroups/myrg/providers/Microsoft.Storage/storageAccounts/mystorageaccount
```

**Remediation:**
```bash
az monitor diagnostic-settings create --name diagnostics --resource /subscriptions/.../storageAccounts/mystorageaccount --logs '[{"category": "StorageRead", "enabled": true}]' --metrics '[{"category": "Transaction", "enabled": true}]' --workspace /subscriptions/.../workspaces/mylogworkspace
```

### 4. Over-privileged Service Principals

**Issue:** Service accounts with excessive permissions
**Risk:** Privilege escalation if compromised
**Detection:**
```bash
az role assignment list --assignee "service-principal-id" --query "[].{role:roleDefinitionName, scope:scope}" -o table
```

**Remediation:**
```bash
az role assignment delete --assignee "service-principal-id" --role "Contributor" --scope "/subscriptions/subscription-id"
```

### 5. Unencrypted Managed Disks

**Issue:** VM disks without encryption
**Risk:** Data exposure if disks are compromised
**Detection:**
```bash
az disk list --resource-group myrg --query "[].{name:name, encryption:encryption.type}" -o table
```

**Remediation:**
```bash
az disk encryption-set create --name mydiskencryptionset --resource-group myrg --key-url "https://mykeyvault.vault.azure.net/keys/mykey" --source-vault mykeyvault
```

## Identity and Access Management

### Azure AD Security Best Practices

1. **Enable Multi-Factor Authentication (MFA)**
```bash
az ad user update --id user@domain.com --force-change-password-next-login true
```

2. **Implement Conditional Access Policies**
```bash
az rest --method PUT --uri "https://graph.microsoft.com/v1.0/identity/conditionalAccess/policies" --body @conditional-access-policy.json
```

3. **Use Managed Identities**
```bash
az vm identity assign --resource-group myrg --name myvm
```

## Monitoring and Alerting

### Enable Azure Security Center
```bash
az security pricing create --name "VirtualMachines" --tier "Standard"
az security pricing create --name "StorageAccounts" --tier "Standard"
```

### Create Security Alerts
```bash
az monitor metrics alert create --name "HighCPUAlert" --resource /subscriptions/.../virtualMachines/myvm --condition "CPU Percentage > 80" --action /subscriptions/.../actionGroups/myactiongroup
```

## Lab Cleanup Script

```bash
python cloud_lab/cleanup_azure_lab.py
```

This will remove all created resources to avoid ongoing costs.

## Security Assessment Checklist

- [ ] Storage accounts have public access disabled
- [ ] NSG rules restrict unnecessary inbound traffic
- [ ] Azure Monitor diagnostics are enabled
- [ ] Service principals follow least privilege
- [ ] Disk encryption is enabled
- [ ] MFA is enforced for privileged accounts
- [ ] Security Center is configured
- [ ] Resource locks prevent accidental deletion
- [ ] Tags are applied for resource management

## Common Azure Security Misconfigurations

1. **Public Storage Containers**
2. **Open Management Ports (3389, 22)**
3. **Disabled Logging and Monitoring**
4. **Over-permissive IAM Roles**
5. **Unencrypted Data at Rest**
6. **Missing Network Security Groups**
7. **Default Passwords and Keys**
8. **Unrestricted API Access**

## Remediation Scripts

See `cloud_lab/remediation_scripts/` for automated fixes for common issues.

## Best Practices Summary

1. **Defense in Depth:** Multiple layers of security controls
2. **Least Privilege:** Minimal required permissions
3. **Zero Trust:** Never trust, always verify
4. **Monitoring:** Continuous security monitoring
5. **Automation:** Infrastructure as Code for consistency
6. **Regular Audits:** Periodic security assessments

## Conclusion

This lab demonstrates the importance of proper cloud security configuration. Azure provides robust security tools, but misconfigurations can lead to significant security risks. Regular security assessments and automated remediation are essential for maintaining a secure cloud environment.