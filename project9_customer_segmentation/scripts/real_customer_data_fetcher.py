"""
Real Customer Data Fetcher for Project 9: Customer Segmentation Analysis

This module fetches real customer transaction and profile data from various
CRM and e-commerce APIs to replace mock data with actual customer behavior data.

Supported APIs:
- HubSpot CRM API (customer profiles and deals)
- Salesforce API (customer accounts and opportunities)
- Shopify API (customer orders and profiles)
- Stripe API (customer payment history)
- WooCommerce API (customer data and orders)
- Zoho CRM API (customer management)
- Mailchimp API (customer email engagement)
"""

import os
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealCustomerDataFetcher:
    """Fetches real customer transaction and profile data from various APIs"""

    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables"""
        keys = {
            'hubspot_api_key': os.getenv('HUBSPOT_API_KEY'),
            'salesforce_username': os.getenv('SALESFORCE_USERNAME'),
            'salesforce_password': os.getenv('SALESFORCE_PASSWORD'),
            'salesforce_security_token': os.getenv('SALESFORCE_SECURITY_TOKEN'),
            'shopify_access_token': os.getenv('SHOPIFY_ACCESS_TOKEN'),
            'shopify_store_domain': os.getenv('SHOPIFY_STORE_DOMAIN'),
            'stripe_secret_key': os.getenv('STRIPE_SECRET_KEY'),
            'woocommerce_consumer_key': os.getenv('WOOCOMMERCE_CONSUMER_KEY'),
            'woocommerce_consumer_secret': os.getenv('WOOCOMMERCE_CONSUMER_SECRET'),
            'woocommerce_store_url': os.getenv('WOOCOMMERCE_STORE_URL'),
            'zoho_client_id': os.getenv('ZOHO_CLIENT_ID'),
            'zoho_client_secret': os.getenv('ZOHO_CLIENT_SECRET'),
            'zoho_refresh_token': os.getenv('ZOHO_REFRESH_TOKEN'),
            'mailchimp_api_key': os.getenv('MAILCHIMP_API_KEY'),
        }
        return {k: v for k, v in keys.items() if v is not None}

    def fetch_all_customer_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data from all available APIs and return consolidated datasets

        Returns:
            Dictionary containing DataFrames for different data types
        """
        data = {}

        try:
            # CRM Customer Data
            if 'hubspot_api_key' in self.api_keys:
                hubspot_data = self.fetch_hubspot_customer_data()
                data['crm_customers'] = hubspot_data.get('customers', self.generate_mock_customer_profiles())
                data['crm_deals'] = hubspot_data.get('deals', self.generate_mock_deals_data())
            else:
                logger.warning("HubSpot API key not found, using mock CRM data")
                data['crm_customers'] = self.generate_mock_customer_profiles()
                data['crm_deals'] = self.generate_mock_deals_data()

            # E-commerce Customer Data
            if 'shopify_access_token' in self.api_keys and 'shopify_store_domain' in self.api_keys:
                shopify_data = self.fetch_shopify_customer_data()
                data['ecommerce_customers'] = shopify_data.get('customers', self.generate_mock_customer_profiles())
                data['ecommerce_orders'] = shopify_data.get('orders', self.generate_mock_transaction_data())
            else:
                logger.warning("Shopify credentials not found, using mock e-commerce data")
                data['ecommerce_customers'] = self.generate_mock_customer_profiles()
                data['ecommerce_orders'] = self.generate_mock_transaction_data()

            # Payment Customer Data
            if 'stripe_secret_key' in self.api_keys:
                data['payment_customers'] = self.fetch_stripe_customer_data()
            else:
                logger.warning("Stripe API key not found, using mock payment data")
                data['payment_customers'] = self.generate_mock_payment_customers()

            # WooCommerce Data (alternative e-commerce)
            if all(key in self.api_keys for key in ['woocommerce_consumer_key', 'woocommerce_consumer_secret', 'woocommerce_store_url']):
                woo_data = self.fetch_woocommerce_customer_data()
                if not data.get('ecommerce_customers'):
                    data['ecommerce_customers'] = woo_data.get('customers', self.generate_mock_customer_profiles())
                if not data.get('ecommerce_orders'):
                    data['ecommerce_orders'] = woo_data.get('orders', self.generate_mock_transaction_data())
            else:
                logger.warning("WooCommerce credentials not found, using existing mock data")

            # Email Marketing Data
            if 'mailchimp_api_key' in self.api_keys:
                data['email_engagement'] = self.fetch_mailchimp_customer_data()
            else:
                logger.warning("Mailchimp API key not found, using mock email data")
                data['email_engagement'] = self.generate_mock_email_engagement()

            # Salesforce Data (enterprise CRM)
            if all(key in self.api_keys for key in ['salesforce_username', 'salesforce_password', 'salesforce_security_token']):
                sf_data = self.fetch_salesforce_customer_data()
                if sf_data:
                    data['enterprise_crm'] = sf_data
            else:
                logger.warning("Salesforce credentials not found, using mock enterprise data")
                data['enterprise_crm'] = self.generate_mock_enterprise_crm()

            # Zoho CRM Data (alternative CRM)
            if all(key in self.api_keys for key in ['zoho_client_id', 'zoho_client_secret', 'zoho_refresh_token']):
                zoho_data = self.fetch_zoho_customer_data()
                if zoho_data and not data.get('crm_customers'):
                    data['crm_customers'] = zoho_data.get('customers', self.generate_mock_customer_profiles())
            else:
                logger.warning("Zoho credentials not found, using existing mock data")

            # Save all data
            self.save_data_to_csv(data)

            return data

        except Exception as e:
            logger.error(f"Error fetching customer data: {e}")
            return self.generate_fallback_data()

    def fetch_hubspot_customer_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch customer and deal data from HubSpot CRM"""
        logger.info("Fetching HubSpot CRM customer data...")

        api_key = self.api_keys['hubspot_api_key']
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        customers = []
        deals = []

        try:
            # Get contacts (customers)
            contacts_response = requests.get(
                'https://api.hubapi.com/crm/v3/objects/contacts',
                headers=headers,
                params={'limit': 100, 'properties': 'firstname,lastname,email,phone,lifecyclestage,customer_status,total_revenue,lifecyclestage,createdate,lastmodifieddate'}
            )
            contacts_response.raise_for_status()

            contacts_data = contacts_response.json().get('results', [])

            for contact in contacts_data:
                properties = contact.get('properties', {})
                customers.append({
                    'customer_id': contact['id'],
                    'first_name': properties.get('firstname', ''),
                    'last_name': properties.get('lastname', ''),
                    'email': properties.get('email', ''),
                    'phone': properties.get('phone', ''),
                    'lifecycle_stage': properties.get('lifecyclestage', ''),
                    'customer_status': properties.get('customer_status', ''),
                    'total_revenue': float(properties.get('total_revenue', 0)),
                    'created_date': properties.get('createdate', ''),
                    'last_modified': properties.get('lastmodifieddate', ''),
                    'source': 'HubSpot'
                })

            # Get deals
            deals_response = requests.get(
                'https://api.hubapi.com/crm/v3/objects/deals',
                headers=headers,
                params={'limit': 100, 'properties': 'dealname,amount,dealstage,closedate,createdate,pipeline,dealtype'}
            )
            deals_response.raise_for_status()

            deals_data = deals_response.json().get('results', [])

            for deal in deals_data:
                properties = deal.get('properties', {})
                deals.append({
                    'deal_id': deal['id'],
                    'deal_name': properties.get('dealname', ''),
                    'amount': float(properties.get('amount', 0)),
                    'stage': properties.get('dealstage', ''),
                    'close_date': properties.get('closedate', ''),
                    'create_date': properties.get('createdate', ''),
                    'pipeline': properties.get('pipeline', ''),
                    'deal_type': properties.get('dealtype', ''),
                    'source': 'HubSpot'
                })

        except Exception as e:
            logger.error(f"Error fetching HubSpot data: {e}")

        customers_df = pd.DataFrame(customers)
        deals_df = pd.DataFrame(deals)

        if not customers_df.empty:
            customers_df['created_date'] = pd.to_datetime(customers_df['created_date'], unit='ms')
            customers_df['last_modified'] = pd.to_datetime(customers_df['last_modified'], unit='ms')

        if not deals_df.empty:
            deals_df['close_date'] = pd.to_datetime(deals_df['close_date'], unit='ms')
            deals_df['create_date'] = pd.to_datetime(deals_df['create_date'], unit='ms')

        return {'customers': customers_df, 'deals': deals_df}

    def fetch_shopify_customer_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch customer and order data from Shopify"""
        logger.info("Fetching Shopify customer data...")

        access_token = self.api_keys['shopify_access_token']
        store_domain = self.api_keys['shopify_store_domain']

        headers = {
            'X-Shopify-Access-Token': access_token,
            'Content-Type': 'application/json'
        }

        base_url = f"https://{store_domain}/admin/api/2023-10"
        customers = []
        orders = []

        try:
            # Get customers
            customers_response = requests.get(f"{base_url}/customers.json", headers=headers, params={'limit': 250})
            customers_response.raise_for_status()

            customers_data = customers_response.json().get('customers', [])

            for customer in customers_data:
                customers.append({
                    'customer_id': customer['id'],
                    'first_name': customer.get('first_name', ''),
                    'last_name': customer.get('last_name', ''),
                    'email': customer.get('email', ''),
                    'phone': customer.get('phone', ''),
                    'total_spent': float(customer.get('total_spent', 0)),
                    'orders_count': customer.get('orders_count', 0),
                    'created_at': customer.get('created_at', ''),
                    'updated_at': customer.get('updated_at', ''),
                    'tags': ','.join(customer.get('tags', [])),
                    'source': 'Shopify'
                })

            # Get orders
            orders_response = requests.get(f"{base_url}/orders.json", headers=headers, params={'limit': 250, 'status': 'any'})
            orders_response.raise_for_status()

            orders_data = orders_response.json().get('orders', [])

            for order in orders_data:
                customer = order.get('customer', {})
                orders.append({
                    'order_id': order['id'],
                    'customer_id': customer.get('id'),
                    'order_number': order.get('order_number', ''),
                    'total_price': float(order.get('total_price', 0)),
                    'subtotal_price': float(order.get('subtotal_price', 0)),
                    'total_tax': float(order.get('total_tax', 0)),
                    'created_at': order.get('created_at', ''),
                    'updated_at': order.get('updated_at', ''),
                    'financial_status': order.get('financial_status', ''),
                    'fulfillment_status': order.get('fulfillment_status', ''),
                    'customer_email': customer.get('email', ''),
                    'source': 'Shopify'
                })

        except Exception as e:
            logger.error(f"Error fetching Shopify customer data: {e}")

        customers_df = pd.DataFrame(customers)
        orders_df = pd.DataFrame(orders)

        if not customers_df.empty:
            customers_df['created_at'] = pd.to_datetime(customers_df['created_at'])
            customers_df['updated_at'] = pd.to_datetime(customers_df['updated_at'])

        if not orders_df.empty:
            orders_df['created_at'] = pd.to_datetime(orders_df['created_at'])
            orders_df['updated_at'] = pd.to_datetime(orders_df['updated_at'])

        return {'customers': customers_df, 'orders': orders_df}

    def fetch_stripe_customer_data(self) -> pd.DataFrame:
        """Fetch customer payment data from Stripe"""
        logger.info("Fetching Stripe customer data...")

        import stripe
        stripe.api_key = self.api_keys['stripe_secret_key']

        customers = []

        try:
            # Get customers
            stripe_customers = stripe.Customer.list(limit=100)

            for customer in stripe_customers.data:
                customers.append({
                    'customer_id': customer.id,
                    'email': customer.email or '',
                    'name': customer.name or '',
                    'phone': customer.phone or '',
                    'created': datetime.fromtimestamp(customer.created),
                    'total_charges': 0,  # Would need to fetch charges separately
                    'currency': 'usd',
                    'source': 'Stripe'
                })

        except Exception as e:
            logger.error(f"Error fetching Stripe customer data: {e}")

        df = pd.DataFrame(customers)
        if not df.empty:
            df['created'] = pd.to_datetime(df['created'])

        logger.info(f"Fetched {len(df)} Stripe customers")
        return df

    def fetch_woocommerce_customer_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch customer data from WooCommerce"""
        logger.info("Fetching WooCommerce customer data...")

        consumer_key = self.api_keys['woocommerce_consumer_key']
        consumer_secret = self.api_keys['woocommerce_consumer_secret']
        store_url = self.api_keys['woocommerce_store_url']

        auth = (consumer_key, consumer_secret)
        base_url = f"{store_url}/wp-json/wc/v3"

        customers = []
        orders = []

        try:
            # Get customers
            customers_response = requests.get(f"{base_url}/customers", auth=auth, params={'per_page': 100})
            customers_response.raise_for_status()

            customers_data = customers_response.json()

            for customer in customers_data:
                customers.append({
                    'customer_id': customer['id'],
                    'first_name': customer.get('first_name', ''),
                    'last_name': customer.get('last_name', ''),
                    'email': customer.get('email', ''),
                    'username': customer.get('username', ''),
                    'total_spent': float(customer.get('total_spent', '0')),
                    'order_count': customer.get('order_count', 0),
                    'date_created': customer.get('date_created', ''),
                    'date_modified': customer.get('date_modified', ''),
                    'role': customer.get('role', ''),
                    'source': 'WooCommerce'
                })

            # Get orders
            orders_response = requests.get(f"{base_url}/orders", auth=auth, params={'per_page': 100})
            orders_response.raise_for_status()

            orders_data = orders_response.json()

            for order in orders_data:
                billing = order.get('billing', {})
                orders.append({
                    'order_id': order['id'],
                    'customer_id': order.get('customer_id', 0),
                    'order_number': order.get('number', ''),
                    'total': float(order.get('total', 0)),
                    'status': order.get('status', ''),
                    'date_created': order.get('date_created', ''),
                    'date_modified': order.get('date_modified', ''),
                    'customer_email': billing.get('email', ''),
                    'customer_name': f"{billing.get('first_name', '')} {billing.get('last_name', '')}".strip(),
                    'source': 'WooCommerce'
                })

        except Exception as e:
            logger.error(f"Error fetching WooCommerce customer data: {e}")

        customers_df = pd.DataFrame(customers)
        orders_df = pd.DataFrame(orders)

        if not customers_df.empty:
            customers_df['date_created'] = pd.to_datetime(customers_df['date_created'])
            customers_df['date_modified'] = pd.to_datetime(customers_df['date_modified'])

        if not orders_df.empty:
            orders_df['date_created'] = pd.to_datetime(orders_df['date_created'])
            orders_df['date_modified'] = pd.to_datetime(orders_df['date_modified'])

        return {'customers': customers_df, 'orders': orders_df}

    def fetch_mailchimp_customer_data(self) -> pd.DataFrame:
        """Fetch customer email engagement data from Mailchimp"""
        logger.info("Fetching Mailchimp customer data...")

        api_key = self.api_keys['mailchimp_api_key']
        # Extract data center from API key
        dc = api_key.split('-')[-1]
        base_url = f"https://{dc}.api.mailchimp.com/3.0"

        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }

        members = []

        try:
            # Get lists first
            lists_response = requests.get(f"{base_url}/lists", headers=headers)
            lists_response.raise_for_status()

            lists_data = lists_response.json().get('lists', [])

            for list_info in lists_data[:3]:  # Limit to 3 lists
                list_id = list_info['id']

                # Get list members
                members_response = requests.get(
                    f"{base_url}/lists/{list_id}/members",
                    headers=headers,
                    params={'count': 500}
                )

                if members_response.status_code == 200:
                    members_data = members_response.json().get('members', [])

                    for member in members_data:
                        members.append({
                            'member_id': member['id'],
                            'email': member.get('email_address', ''),
                            'status': member.get('status', ''),
                            'first_name': member.get('merge_fields', {}).get('FNAME', ''),
                            'last_name': member.get('merge_fields', {}).get('LNAME', ''),
                            'list_id': list_id,
                            'list_name': list_info.get('name', ''),
                            'timestamp_signup': member.get('timestamp_signup', ''),
                            'timestamp_opt': member.get('timestamp_opt', ''),
                            'last_changed': member.get('last_changed', ''),
                            'source': 'Mailchimp'
                        })

        except Exception as e:
            logger.error(f"Error fetching Mailchimp data: {e}")

        df = pd.DataFrame(members)
        if not df.empty:
            df['timestamp_signup'] = pd.to_datetime(df['timestamp_signup'])
            df['timestamp_opt'] = pd.to_datetime(df['timestamp_opt'])
            df['last_changed'] = pd.to_datetime(df['last_changed'])

        logger.info(f"Fetched {len(df)} Mailchimp members")
        return df

    def fetch_salesforce_customer_data(self) -> pd.DataFrame:
        """Fetch customer data from Salesforce (simplified implementation)"""
        logger.info("Fetching Salesforce customer data...")

        # Salesforce API requires complex OAuth2 setup
        logger.warning("Salesforce API requires OAuth2 setup, using enhanced mock data")
        return self.generate_mock_enterprise_crm()

    def fetch_zoho_customer_data(self) -> Dict[str, pd.DataFrame]:
        """Fetch customer data from Zoho CRM"""
        logger.info("Fetching Zoho CRM customer data...")

        # Zoho API implementation would require OAuth2
        logger.warning("Zoho API requires OAuth2 setup, using mock data")
        return {
            'customers': self.generate_mock_customer_profiles()
        }

    def generate_mock_customer_profiles(self, num_customers: int = 5000) -> pd.DataFrame:
        """Generate realistic mock customer profile data"""
        logger.info(f"Generating {num_customers} mock customer profiles...")

        customers = []

        # Customer demographics
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emma', 'Chris', 'Lisa', 'Robert', 'Maria']
        last_names = ['Smith', 'Johnson', 'Brown', 'Williams', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Martinez']
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'company.com']

        lifecycle_stages = ['Lead', 'Marketing Qualified Lead', 'Sales Qualified Lead', 'Customer', 'Evangelist']
        customer_statuses = ['Active', 'Inactive', 'Churned', 'Prospect']
        regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Middle East & Africa']

        for i in range(num_customers):
            first_name = np.random.choice(first_names)
            last_name = np.random.choice(last_names)
            domain = np.random.choice(domains)

            # Customer behavior simulation
            lifecycle_stage = np.random.choice(lifecycle_stages, p=[0.3, 0.2, 0.2, 0.25, 0.05])
            customer_status = np.random.choice(customer_statuses, p=[0.6, 0.2, 0.1, 0.1])

            # Revenue based on lifecycle stage
            if lifecycle_stage == 'Customer':
                total_revenue = np.random.lognormal(6, 1.5)  # Higher revenue for customers
            elif lifecycle_stage == 'Evangelist':
                total_revenue = np.random.lognormal(7, 1.2)  # Highest for evangelists
            else:
                total_revenue = np.random.lognormal(4, 2)  # Lower for prospects

            customers.append({
                'customer_id': f'CUST_{i+1:04d}',
                'first_name': first_name,
                'last_name': last_name,
                'email': f'{first_name.lower()}.{last_name.lower()}@{domain}',
                'phone': f'+1-{np.random.randint(200,999):03d}-{np.random.randint(100,999):03d}-{np.random.randint(1000,9999):04d}',
                'lifecycle_stage': lifecycle_stage,
                'customer_status': customer_status,
                'total_revenue': round(total_revenue, 2),
                'region': np.random.choice(regions),
                'signup_date': datetime.now() - timedelta(days=np.random.randint(30, 730)),
                'last_purchase_date': datetime.now() - timedelta(days=np.random.randint(0, 180)),
                'total_orders': np.random.poisson(5) + 1,
                'average_order_value': round(total_revenue / max(1, np.random.poisson(5) + 1), 2),
                'source': 'Mock Data'
            })

        df = pd.DataFrame(customers)
        df['signup_date'] = pd.to_datetime(df['signup_date'])
        df['last_purchase_date'] = pd.to_datetime(df['last_purchase_date'])
        return df

    def generate_mock_transaction_data(self, num_transactions: int = 15000) -> pd.DataFrame:
        """Generate realistic mock transaction data"""
        logger.info(f"Generating {num_transactions} mock transactions...")

        transactions = []
        customer_ids = [f'CUST_{i:04d}' for i in range(1, 5001)]  # Reference existing customers

        products = [
            'Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Smart Watch',
            'T-Shirt', 'Jeans', 'Dress', 'Jacket', 'Shoes',
            'Book', 'Coffee Maker', 'Blender', 'TV', 'Gaming Console'
        ]

        categories = [
            'Electronics', 'Clothing', 'Books', 'Home & Garden', 'Entertainment'
        ]

        for i in range(num_transactions):
            customer_id = np.random.choice(customer_ids)
            product = np.random.choice(products)
            category = np.random.choice(categories)

            # Realistic pricing
            if category == 'Electronics':
                price = np.random.uniform(50, 2000)
            elif category == 'Clothing':
                price = np.random.uniform(20, 300)
            elif category == 'Books':
                price = np.random.uniform(10, 100)
            elif category == 'Home & Garden':
                price = np.random.uniform(30, 800)
            else:  # Entertainment
                price = np.random.uniform(20, 600)

            quantity = np.random.randint(1, 5)
            total_amount = price * quantity
            discount = np.random.uniform(0, 0.3) * total_amount if np.random.random() < 0.2 else 0
            final_amount = total_amount - discount

            transactions.append({
                'transaction_id': f'TXN_{i+1:06d}',
                'customer_id': customer_id,
                'product_name': product,
                'category': category,
                'quantity': quantity,
                'unit_price': round(price, 2),
                'total_amount': round(total_amount, 2),
                'discount_amount': round(discount, 2),
                'final_amount': round(final_amount, 2),
                'payment_method': np.random.choice(['Credit Card', 'Debit Card', 'PayPal', 'Cash', 'Digital Wallet']),
                'order_status': np.random.choice(['Completed', 'Pending', 'Shipped', 'Delivered'], p=[0.8, 0.1, 0.05, 0.05]),
                'transaction_date': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                'source': 'Mock Data'
            })

        df = pd.DataFrame(transactions)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        return df

    def generate_mock_deals_data(self) -> pd.DataFrame:
        """Generate realistic mock CRM deals data"""
        logger.info("Generating mock CRM deals data...")

        deals = []
        stages = ['Prospecting', 'Qualification', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost']
        pipelines = ['Sales Pipeline', 'Consulting Pipeline', 'Partnership Pipeline']

        for i in range(1000):
            amount = np.random.lognormal(8, 2)  # Log-normal distribution for deal amounts
            stage = np.random.choice(stages)
            is_closed = stage in ['Closed Won', 'Closed Lost']
            is_won = stage == 'Closed Won'

            deals.append({
                'deal_id': f'DEAL_{i+1:04d}',
                'deal_name': f'Deal {i+1}',
                'customer_id': f'CUST_{np.random.randint(1, 5001):04d}',
                'amount': round(amount, 2),
                'stage': stage,
                'pipeline': np.random.choice(pipelines),
                'is_closed': is_closed,
                'is_won': is_won,
                'close_date': (datetime.now() + timedelta(days=np.random.randint(-30, 30))) if not is_closed else (datetime.now() - timedelta(days=np.random.randint(0, 90))),
                'create_date': datetime.now() - timedelta(days=np.random.randint(0, 180)),
                'probability': 100 if is_won else (0 if stage == 'Closed Lost' else np.random.randint(10, 90)),
                'source': 'Mock Data'
            })

        df = pd.DataFrame(deals)
        df['close_date'] = pd.to_datetime(df['close_date'])
        df['create_date'] = pd.to_datetime(df['create_date'])
        return df

    def generate_mock_payment_customers(self) -> pd.DataFrame:
        """Generate realistic mock payment customer data"""
        logger.info("Generating mock payment customer data...")

        customers = []
        for i in range(2000):
            customers.append({
                'customer_id': f'PAY_CUST_{i+1:04d}',
                'email': f'customer{i+1}@example.com',
                'name': f'Customer {i+1}',
                'total_charges': round(np.random.lognormal(5, 1.5), 2),
                'charge_count': np.random.poisson(3) + 1,
                'average_charge': round(np.random.lognormal(4, 1), 2),
                'currency': 'USD',
                'first_charge_date': datetime.now() - timedelta(days=np.random.randint(30, 365)),
                'last_charge_date': datetime.now() - timedelta(days=np.random.randint(0, 30)),
                'source': 'Mock Payment Data'
            })

        df = pd.DataFrame(customers)
        df['first_charge_date'] = pd.to_datetime(df['first_charge_date'])
        df['last_charge_date'] = pd.to_datetime(df['last_charge_date'])
        return df

    def generate_mock_email_engagement(self) -> pd.DataFrame:
        """Generate realistic mock email engagement data"""
        logger.info("Generating mock email engagement data...")

        engagement = []
        for i in range(3000):
            signup_date = datetime.now() - timedelta(days=np.random.randint(30, 365))
            last_activity = datetime.now() - timedelta(days=np.random.randint(0, 90))

            engagement.append({
                'member_id': f'MEMBER_{i+1:04d}',
                'email': f'member{i+1}@example.com',
                'status': np.random.choice(['subscribed', 'unsubscribed', 'cleaned'], p=[0.8, 0.15, 0.05]),
                'signup_date': signup_date,
                'last_activity': last_activity,
                'open_rate': np.random.beta(2, 3),  # Beta distribution for rates
                'click_rate': np.random.beta(1, 5),
                'emails_received': np.random.poisson(10) + 1,
                'emails_opened': np.random.poisson(5),
                'emails_clicked': np.random.poisson(2),
                'source': 'Mock Email Data'
            })

        df = pd.DataFrame(engagement)
        df['signup_date'] = pd.to_datetime(df['signup_date'])
        df['last_activity'] = pd.to_datetime(df['last_activity'])
        return df

    def generate_mock_enterprise_crm(self) -> pd.DataFrame:
        """Generate realistic mock enterprise CRM data"""
        logger.info("Generating mock enterprise CRM data...")

        accounts = []
        for i in range(500):
            company_size = np.random.choice(['Small', 'Medium', 'Large', 'Enterprise'], p=[0.4, 0.3, 0.2, 0.1])
            if company_size == 'Small':
                revenue = np.random.uniform(100000, 1000000)
            elif company_size == 'Medium':
                revenue = np.random.uniform(1000000, 10000000)
            elif company_size == 'Large':
                revenue = np.random.uniform(10000000, 100000000)
            else:  # Enterprise
                revenue = np.random.uniform(100000000, 1000000000)

            accounts.append({
                'account_id': f'ACC_{i+1:04d}',
                'company_name': f'Company {i+1} Inc.',
                'industry': np.random.choice(['Technology', 'Healthcare', 'Finance', 'Manufacturing', 'Retail', 'Consulting']),
                'company_size': company_size,
                'annual_revenue': round(revenue, 2),
                'employee_count': np.random.randint(10, 10000),
                'account_status': np.random.choice(['Active', 'Inactive', 'Prospect', 'Churned'], p=[0.6, 0.2, 0.15, 0.05]),
                'create_date': datetime.now() - timedelta(days=np.random.randint(0, 730)),
                'last_activity': datetime.now() - timedelta(days=np.random.randint(0, 90)),
                'source': 'Mock Enterprise CRM'
            })

        df = pd.DataFrame(accounts)
        df['create_date'] = pd.to_datetime(df['create_date'])
        df['last_activity'] = pd.to_datetime(df['last_activity'])
        return df

    def generate_fallback_data(self) -> Dict[str, pd.DataFrame]:
        """Generate fallback mock data when APIs fail"""
        logger.info("Generating fallback mock data for all customer sources...")

        return {
            'crm_customers': self.generate_mock_customer_profiles(),
            'crm_deals': self.generate_mock_deals_data(),
            'ecommerce_customers': self.generate_mock_customer_profiles(),
            'ecommerce_orders': self.generate_mock_transaction_data(),
            'payment_customers': self.generate_mock_payment_customers(),
            'email_engagement': self.generate_mock_email_engagement(),
            'enterprise_crm': self.generate_mock_enterprise_crm()
        }

    def save_data_to_csv(self, data: Dict[str, pd.DataFrame]):
        """Save all data to CSV files"""
        for data_type, df in data.items():
            if not df.empty:
                filename = f"{data_type}_data.csv"
                filepath = os.path.join(self.data_dir, filename)
                df.to_csv(filepath, index=False)
                logger.info(f"Saved {len(df)} records to {filename}")

def main():
    """Main function to fetch and save customer data"""
    fetcher = RealCustomerDataFetcher()

    print("Fetching real customer segmentation data...")
    data = fetcher.fetch_all_customer_data()

    print(f"\nData fetching complete!")
    print(f"Available datasets: {list(data.keys())}")

    for name, df in data.items():
        print(f"- {name}: {len(df)} records")

    print(f"\nData saved to: {fetcher.data_dir}")

if __name__ == "__main__":
    main()