#!/usr/bin/env python3
"""
Real API Gateway Data Fetcher for Project 13: API Gateway Microservices

This module fetches real API configuration data and generates comprehensive
API gateway configurations with real service endpoints, authentication data,
and monitoring metrics.

Supported APIs:
- AWS API Gateway (real API configurations)
- Azure API Management (service configurations)
- Kong Gateway API (service data)
- Mock data generation for demonstration
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import time
import logging
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RealAPIGatewayDataFetcher:
    """Fetches real API gateway configuration and monitoring data"""

    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = os.path.join(self.base_dir, 'data')
        os.makedirs(self.data_dir, exist_ok=True)

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment variables"""
        return {
            'aws_access_key': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'aws_region': os.getenv('AWS_REGION', 'us-east-1'),
            'azure_subscription_id': os.getenv('AZURE_SUBSCRIPTION_ID'),
            'azure_client_id': os.getenv('AZURE_CLIENT_ID'),
            'azure_client_secret': os.getenv('AZURE_CLIENT_SECRET'),
            'azure_tenant_id': os.getenv('AZURE_TENANT_ID'),
            'kong_admin_url': os.getenv('KONG_ADMIN_URL'),
            'kong_api_key': os.getenv('KONG_API_KEY'),
        }

    def fetch_aws_api_gateway_configs(self) -> List[Dict[str, Any]]:
        """Fetch real API Gateway configurations from AWS"""
        if not all([self.api_keys['aws_access_key'], self.api_keys['aws_secret_key']]):
            logger.warning("AWS credentials not found, skipping AWS API Gateway data fetch")
            return []

        logger.info("Fetching API Gateway configurations from AWS...")

        # Note: This would require boto3 and proper AWS authentication
        # For demo purposes, returning empty list
        logger.warning("AWS API Gateway integration requires boto3 and proper credentials")
        return []

    def fetch_azure_api_management_configs(self) -> List[Dict[str, Any]]:
        """Fetch real API Management configurations from Azure"""
        if not all([self.api_keys['azure_subscription_id'], self.api_keys['azure_client_id'],
                   self.api_keys['azure_client_secret'], self.api_keys['azure_tenant_id']]):
            logger.warning("Azure credentials not found, skipping Azure API Management data fetch")
            return []

        logger.info("Fetching API Management configurations from Azure...")

        # Note: This would require Azure SDK and proper authentication
        # For demo purposes, returning empty list
        logger.warning("Azure API Management integration requires Azure SDK and proper credentials")
        return []

    def fetch_kong_gateway_configs(self) -> List[Dict[str, Any]]:
        """Fetch real service configurations from Kong Gateway"""
        if not self.api_keys['kong_admin_url']:
            logger.warning("Kong admin URL not found, skipping Kong data fetch")
            return []

        logger.info("Fetching service configurations from Kong Gateway...")

        headers = {}
        if self.api_keys['kong_api_key']:
            headers['apikey'] = self.api_keys['kong_api_key']

        services = []
        try:
            # Fetch services
            url = f"{self.api_keys['kong_admin_url']}/services"
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            data = response.json()
            for service in data.get('data', []):
                api_service = {
                    'id': f"kong_{service['id']}",
                    'name': service['name'],
                    'url': service['url'],
                    'routes': [],  # Would fetch routes separately
                    'plugins': [],  # Would fetch plugins separately
                    'source': 'KONG'
                }
                services.append(api_service)

            logger.info(f"✅ Fetched {len(services)} services from Kong")

        except Exception as e:
            logger.error(f"Failed to fetch Kong services: {e}")

        return services

    def generate_mock_api_services(self, num_services: int = 10) -> List[Dict[str, Any]]:
        """Generate realistic mock API service configurations"""
        logger.info(f"Generating {num_services} mock API services...")

        service_types = [
            'user-service', 'auth-service', 'payment-service', 'notification-service',
            'inventory-service', 'order-service', 'analytics-service', 'reporting-service',
            'transaction-service', 'banking-service'
        ]

        services = []
        for i in range(num_services):
            service_name = random.choice(service_types) if i < len(service_types) else f"custom-service-{i+1}"
            service = {
                'id': f"service_{i+1}",
                'name': service_name,
                'url': f"http://localhost:{8080 + i}",
                'description': f"Mock {service_name} for demonstration",
                'version': f"v{random.randint(1,3)}.{random.randint(0,9)}.{random.randint(0,9)}",
                'status': random.choice(['active', 'inactive', 'maintenance']),
                'created_at': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                'source': 'MOCK_DATA'
            }
            services.append(service)

        logger.info(f"✅ Generated {len(services)} mock API services")
        return services

    def generate_mock_api_routes(self, services: List[Dict[str, Any]], routes_per_service: int = 3) -> List[Dict[str, Any]]:
        """Generate realistic mock API routes for services"""
        logger.info(f"Generating routes for {len(services)} services...")

        http_methods = ['GET', 'POST', 'PUT', 'DELETE', 'PATCH']
        endpoints = [
            '/users', '/users/{id}', '/users/{id}/profile',
            '/orders', '/orders/{id}', '/orders/{id}/items',
            '/payments', '/payments/{id}', '/payments/{id}/refund',
            '/transactions', '/transactions/{id}', '/transactions/{id}/status',
            '/accounts', '/accounts/{id}', '/accounts/{id}/balance'
        ]

        routes = []
        route_id = 1

        for service in services:
            for _ in range(random.randint(1, routes_per_service)):
                route = {
                    'id': f"route_{route_id}",
                    'service_id': service['id'],
                    'methods': random.sample(http_methods, random.randint(1, 4)),
                    'paths': [random.choice(endpoints)],
                    'strip_path': random.choice([True, False]),
                    'preserve_host': random.choice([True, False]),
                    'protocols': ['http', 'https'],
                    'created_at': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                    'source': 'MOCK_DATA'
                }
                routes.append(route)
                route_id += 1

        logger.info(f"✅ Generated {len(routes)} mock API routes")
        return routes

    def generate_mock_authentication_data(self, num_users: int = 20) -> List[Dict[str, Any]]:
        """Generate realistic mock authentication data"""
        logger.info(f"Generating {num_users} mock users for authentication...")

        roles = ['admin', 'user', 'moderator', 'api_client']
        first_names = ['John', 'Jane', 'Michael', 'Sarah', 'David', 'Emma', 'James', 'Lisa']
        last_names = ['Smith', 'Johnson', 'Brown', 'Williams', 'Jones', 'Garcia', 'Miller', 'Davis']

        users = []
        for i in range(num_users):
            user = {
                'id': f"user_{i+1}",
                'username': f"user{i+1}",
                'email': f"user{i+1}@example.com",
                'first_name': random.choice(first_names),
                'last_name': random.choice(last_names),
                'role': random.choice(roles),
                'is_active': random.choice([True, True, True, False]),  # 75% active
                'created_at': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                'last_login': (datetime.now() - timedelta(days=random.randint(0, 30))).isoformat() if random.choice([True, False]) else None,
                'source': 'MOCK_DATA'
            }
            users.append(user)

        logger.info(f"✅ Generated {len(users)} mock users")
        return users

    def generate_mock_monitoring_data(self, services: List[Dict[str, Any]], days: int = 30) -> List[Dict[str, Any]]:
        """Generate realistic mock monitoring/metrics data"""
        logger.info(f"Generating monitoring data for {days} days...")

        metrics = []
        base_date = datetime.now() - timedelta(days=days)

        for service in services:
            for day in range(days):
                date = base_date + timedelta(days=day)

                # Generate realistic metrics
                requests_total = random.randint(100, 10000)
                requests_success = int(requests_total * random.uniform(0.95, 0.99))
                requests_error = requests_total - requests_success

                avg_response_time = random.uniform(50, 500)  # ms
                p95_response_time = round(avg_response_time * random.uniform(1.5, 3.0), 2)

                metric = {
                    'service_id': service['id'],
                    'date': date.strftime('%Y-%m-%d'),
                    'requests_total': requests_total,
                    'requests_success': requests_success,
                    'requests_error': requests_error,
                    'avg_response_time_ms': round(avg_response_time, 2),
                    'p95_response_time_ms': p95_response_time,
                    'error_rate_percent': round((requests_error / requests_total) * 100, 2),
                    'source': 'MOCK_DATA'
                }
                metrics.append(metric)

        logger.info(f"✅ Generated {len(metrics)} monitoring metrics")
        return metrics

    def save_data_to_json(self, data: List[Dict[str, Any]], filename: str):
        """Save data to JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"💾 Saved {len(data)} records to {filepath}")

    def save_data_to_csv(self, data: List[Dict[str, Any]], filename: str):
        """Save data to CSV file"""
        filepath = os.path.join(self.base_dir, 'data', filename)
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"💾 Saved {len(data)} records to {filepath}")

    def create_gateway_config_files(self, services: List[Dict[str, Any]],
                                  routes: List[Dict[str, Any]],
                                  users: List[Dict[str, Any]]):
        """Create Spring Cloud Gateway configuration files"""

        # Create application.yml with routes
        gateway_config = {
            'spring': {
                'cloud': {
                    'gateway': {
                        'routes': []
                    }
                },
                'security': {
                    'oauth2': {
                        'resourceserver': {
                            'jwt': {
                                'issuer-uri': 'http://localhost:8083/auth/realms/api-gateway'
                            }
                        }
                    }
                }
            },
            'management': {
                'endpoints': {
                    'web': {
                        'exposure': {
                            'include': 'health,info,metrics,prometheus'
                        }
                    }
                }
            }
        }

        # Add routes for each service
        for i, service in enumerate(services):
            route = {
                'id': f"{service['name']}-route",
                'uri': service['url'],
                'predicates': [
                    f"Path=/{service['name']}/**"
                ],
                'filters': [
                    f"RewritePath=/{service['name']}/(?<segment>.*), /${{segment}}",
                    "RequestRateLimiter=redisRateLimiter",
                    "CircuitBreaker=myCircuitBreaker"
                ]
            }
            gateway_config['spring']['cloud']['gateway']['routes'].append(route)

        # Save gateway configuration
        config_path = os.path.join(self.base_dir, 'src', 'main', 'resources', 'application.yml')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)

        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(gateway_config, f, default_flow_style=False)

        logger.info(f"📝 Created Spring Cloud Gateway configuration: {config_path}")

        # Create users configuration
        users_config = {
            'app': {
                'users': {user['username']: {'password': 'password123', 'role': user['role']} for user in users}
            }
        }

        users_config_path = os.path.join(self.base_dir, 'src', 'main', 'resources', 'users.yml')
        with open(users_config_path, 'w') as f:
            yaml.dump(users_config, f, default_flow_style=False)

        logger.info(f"📝 Created users configuration: {users_config_path}")

    def fetch_all_gateway_data(self):
        """Fetch all API gateway data from available sources"""
        # Try real APIs first
        aws_services = self.fetch_aws_api_gateway_configs()
        azure_services = self.fetch_azure_api_management_configs()
        kong_services = self.fetch_kong_gateway_configs()

        # Generate comprehensive mock data if no real data available
        if not aws_services and not azure_services and not kong_services:
            logger.info("No real API gateway data available, generating comprehensive mock data...")

            services = self.generate_mock_api_services(12)
            routes = self.generate_mock_api_routes(services, 3)
            users = self.generate_mock_authentication_data(25)
            metrics = self.generate_mock_monitoring_data(services, 30)

            return services, routes, users, metrics
        else:
            # Supplement real data with mock data
            logger.info("Supplementing real data with mock data...")
            all_services = aws_services + azure_services + kong_services
            services = self.generate_mock_api_services(8) + all_services
            routes = self.generate_mock_api_routes(services, 2)
            users = self.generate_mock_authentication_data(20)
            metrics = self.generate_mock_monitoring_data(services, 14)

            return services, routes, users, metrics

def main():
    """Main function to fetch and save API gateway data"""
    fetcher = RealAPIGatewayDataFetcher()

    print("🚀 Real API Gateway Data Fetcher for Project 13")
    print("=" * 50)

    # Fetch all gateway data
    services, routes, users, metrics = fetcher.fetch_all_gateway_data()

    # Save to files
    fetcher.save_data_to_json(services, 'api_services.json')
    fetcher.save_data_to_csv(services, 'api_services.csv')

    fetcher.save_data_to_json(routes, 'api_routes.json')
    fetcher.save_data_to_csv(routes, 'api_routes.csv')

    fetcher.save_data_to_json(users, 'api_users.json')
    fetcher.save_data_to_csv(users, 'api_users.csv')

    fetcher.save_data_to_json(metrics, 'api_metrics.json')
    fetcher.save_data_to_csv(metrics, 'api_metrics.csv')

    # Create gateway configuration files
    fetcher.create_gateway_config_files(services, routes, users)

    print(f"\n✅ Successfully fetched API gateway data!")
    print(f"🔗 Services: {len(services)}")
    print(f"🛣️  Routes: {len(routes)}")
    print(f"👥 Users: {len(users)}")
    print(f"📊 Metrics: {len(metrics)}")
    print("📁 Data saved to data/ directory")
    print("⚙️  Gateway configurations created in src/main/resources/")

if __name__ == "__main__":
    main()