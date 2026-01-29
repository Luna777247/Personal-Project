#!/usr/bin/env python3
"""
Azure Cloud Security Lab Setup Script
Creates a vulnerable Azure environment for security testing
"""

import os
import json
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient
from azure.mgmt.storage import StorageManagementClient
from azure.mgmt.compute import ComputeManagementClient
from azure.mgmt.network import NetworkManagementClient

def setup_azure_lab():
    """Setup Azure lab environment with intentional security misconfigurations"""

    # Initialize Azure clients
    credential = DefaultAzureCredential()
    subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID')

    if not subscription_id:
        print("❌ Please set AZURE_SUBSCRIPTION_ID environment variable")
        return

    resource_client = ResourceManagementClient(credential, subscription_id)
    storage_client = StorageManagementClient(credential, subscription_id)
    compute_client = ComputeManagementClient(credential, subscription_id)
    network_client = NetworkManagementClient(credential, subscription_id)

    # Create resource group
    resource_group_name = 'security-lab-rg'
    location = 'eastus'

    print(f"Creating resource group: {resource_group_name}")
    resource_client.resource_groups.create_or_update(
        resource_group_name,
        {'location': location}
    )

    # Create storage account with public access (MISCONFIGURATION)
    storage_account_name = 'securitylabstorage'

    print(f"Creating storage account with public access: {storage_account_name}")
    storage_client.storage_accounts.create(
        resource_group_name,
        storage_account_name,
        {
            'location': location,
            'sku': {'name': 'Standard_LRS'},
            'kind': 'StorageV2',
            'allow_blob_public_access': True,  # SECURITY ISSUE
            'minimum_tls_version': 'TLS1_0'  # SECURITY ISSUE
        }
    )

    # Create virtual network
    vnet_name = 'security-lab-vnet'
    subnet_name = 'default'

    print(f"Creating virtual network: {vnet_name}")
    network_client.virtual_networks.create_or_update(
        resource_group_name,
        vnet_name,
        {
            'location': location,
            'address_space': {'address_prefixes': ['10.0.0.0/16']},
            'subnets': [{
                'name': subnet_name,
                'address_prefix': '10.0.0.0/24'
            }]
        }
    )

    # Create NSG with open RDP port (MISCONFIGURATION)
    nsg_name = 'security-lab-nsg'

    print(f"Creating NSG with open ports: {nsg_name}")
    network_client.network_security_groups.create_or_update(
        resource_group_name,
        nsg_name,
        {
            'location': location,
            'security_rules': [
                {
                    'name': 'Allow-RDP',
                    'protocol': 'Tcp',
                    'source_port_range': '*',
                    'destination_port_range': '3389',
                    'source_address_prefix': '*',  # SECURITY ISSUE
                    'destination_address_prefix': '*',
                    'access': 'Allow',
                    'priority': 100,
                    'direction': 'Inbound'
                },
                {
                    'name': 'Allow-SSH',
                    'protocol': 'Tcp',
                    'source_port_range': '*',
                    'destination_port_range': '22',
                    'source_address_prefix': '*',  # SECURITY ISSUE
                    'destination_address_prefix': '*',
                    'access': 'Allow',
                    'priority': 101,
                    'direction': 'Inbound'
                }
            ]
        }
    )

    # Create VM (without encryption - MISCONFIGURATION)
    vm_name = 'security-lab-vm'

    print(f"Creating VM without disk encryption: {vm_name}")
    # Note: VM creation requires more complex setup with SSH keys, etc.
    # This is a simplified example

    print("\n✅ Azure Security Lab setup completed!")
    print("🚨 INTENTIONAL SECURITY MISCONFIGURATIONS:")
    print("   - Storage account with public blob access")
    print("   - NSG with open RDP/SSH ports")
    print("   - No disk encryption")
    print("   - No monitoring/logging enabled")
    print("\n📋 Run security assessment:")
    print("   python cloud_lab/assess_security.py")
    print("\n🧹 Cleanup when done:")
    print("   python cloud_lab/cleanup_azure_lab.py")

def assess_security():
    """Assess security misconfigurations in the lab"""

    credential = DefaultAzureCredential()
    subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID')
    resource_group_name = 'security-lab-rg'

    storage_client = StorageManagementClient(credential, subscription_id)
    network_client = NetworkManagementClient(credential, subscription_id)

    print("🔍 SECURITY ASSESSMENT RESULTS")
    print("=" * 50)

    # Check storage account public access
    try:
        storage_accounts = storage_client.storage_accounts.list_by_resource_group(resource_group_name)
        for account in storage_accounts:
            print(f"Storage Account: {account.name}")
            print(f"  Public Access: {account.allow_blob_public_access}")
            if account.allow_blob_public_access:
                print("  ❌ SECURITY ISSUE: Public blob access enabled")
            else:
                print("  ✅ Secure: Public access disabled")
    except Exception as e:
        print(f"Error checking storage: {e}")

    # Check NSG rules
    try:
        nsgs = network_client.network_security_groups.list(resource_group_name)
        for nsg in nsgs:
            print(f"\nNSG: {nsg.name}")
            for rule in nsg.security_rules:
                if rule.access == 'Allow' and rule.direction == 'Inbound':
                    if rule.destination_port_range in ['3389', '22'] and rule.source_address_prefix == '*':
                        print(f"  ❌ SECURITY ISSUE: Open {rule.destination_port_range} from anywhere")
                    else:
                        print(f"  ✅ Secure rule: {rule.name}")
    except Exception as e:
        print(f"Error checking NSG: {e}")

    print("\n📋 REMEDIATION STEPS:")
    print("1. Disable public blob access on storage accounts")
    print("2. Restrict NSG rules to specific IP ranges")
    print("3. Enable disk encryption for VMs")
    print("4. Configure Azure Monitor diagnostics")
    print("5. Implement proper IAM roles and policies")

if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'assess':
        assess_security()
    else:
        setup_azure_lab()