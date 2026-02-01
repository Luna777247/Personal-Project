#!/usr/bin/env python3
"""
Azure Cloud Security Lab Cleanup Script
Removes all resources created by the security lab
"""

import os
from azure.identity import DefaultAzureCredential
from azure.mgmt.resource import ResourceManagementClient

def cleanup_azure_lab():
    """Clean up Azure lab resources"""

    credential = DefaultAzureCredential()
    subscription_id = os.environ.get('AZURE_SUBSCRIPTION_ID')

    if not subscription_id:
        print("❌ Please set AZURE_SUBSCRIPTION_ID environment variable")
        return

    resource_client = ResourceManagementClient(credential, subscription_id)
    resource_group_name = 'security-lab-rg'

    print(f"🧹 Cleaning up Azure Security Lab resources...")
    print(f"Deleting resource group: {resource_group_name}")

    try:
        # Delete entire resource group (includes all resources)
        delete_operation = resource_client.resource_groups.begin_delete(resource_group_name)
        delete_operation.wait()

        print("✅ All lab resources deleted successfully!")
        print("💰 This will prevent ongoing Azure charges")

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        print("You may need to manually delete resources from Azure portal")

if __name__ == '__main__':
    cleanup_azure_lab()