# Transaction Management REST API

A comprehensive REST API for managing users and transactions with JWT authentication, built with Spring Boot and PostgreSQL.

## Features

- **User Management**: CRUD operations for users with account balances
- **Transaction Management**: Full transaction lifecycle management
- **Money Transfers**: Secure internal money transfers between users
- **JWT Authentication**: Secure token-based authentication
- **PostgreSQL Database**: Robust data persistence
- **Spring Security**: Comprehensive security configuration
- **Validation**: Input validation and error handling

## Technology Stack

- **Backend**: Spring Boot 3.2.0
- **Database**: PostgreSQL
- **Security**: Spring Security + JWT
- **ORM**: Spring Data JPA
- **Validation**: Bean Validation
- **Build Tool**: Maven

## Prerequisites

- Java 17 or higher
- PostgreSQL 12 or higher
- Maven 3.6 or higher

## Database Setup

1. Install PostgreSQL and create a database:
```sql
CREATE DATABASE transaction_db;
CREATE USER postgres WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE transaction_db TO postgres;
```

2. Update `src/main/resources/application.properties` with your database credentials.

## Installation & Running

1. Clone the repository:
```bash
git clone <repository-url>
cd transaction-api
```

2. Configure database in `application.properties`

3. Build the project:
```bash
mvn clean install
```

4. Run the application:
```bash
mvn spring-boot:run
```

The API will be available at `http://localhost:8080`

## API Endpoints

### Authentication
- `POST /api/auth/signin` - User login
- `POST /api/auth/signup` - User registration

### Users
- `GET /api/users` - Get all users
- `GET /api/users/{id}` - Get user by ID
- `POST /api/users` - Create new user
- `PUT /api/users/{id}` - Update user
- `DELETE /api/users/{id}` - Delete user
- `GET /api/users/{id}/balance` - Get account balance

### Transactions
- `GET /api/transactions` - Get all transactions
- `GET /api/transactions/{id}` - Get transaction by ID
- `GET /api/transactions/user/{userId}` - Get user's transactions
- `POST /api/transactions` - Create transaction
- `PUT /api/transactions/{id}` - Update transaction
- `DELETE /api/transactions/{id}` - Delete transaction
- `POST /api/transactions/transfer` - Transfer money between users
- `GET /api/transactions/user/{userId}/date-range` - Get transactions by date range

## API Usage Examples

### 1. Register a new user
```bash
curl -X POST http://localhost:8080/api/auth/signup \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "email": "john@example.com",
    "password": "password123",
    "firstName": "John",
    "lastName": "Doe"
  }'
```

### 2. Login
```bash
curl -X POST http://localhost:8080/api/auth/signin \
  -H "Content-Type: application/json" \
  -d '{
    "username": "johndoe",
    "password": "password123"
  }'
```

### 3. Create a transaction (requires JWT token)
```bash
curl -X POST http://localhost:8080/api/transactions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "transactionType": "DEPOSIT",
    "amount": 1000.00,
    "description": "Initial deposit",
    "user": {
      "id": 1
    }
  }'
```

### 4. Transfer money
```bash
curl -X POST http://localhost:8080/api/transactions/transfer \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{
    "senderId": 1,
    "recipientId": 2,
    "amount": 500.00,
    "description": "Payment for services"
  }'
```

## Data Models

### User
```json
{
  "id": 1,
  "username": "johndoe",
  "email": "john@example.com",
  "firstName": "John",
  "lastName": "Doe",
  "accountBalance": 1500.00,
  "createdAt": "2024-01-01T10:00:00",
  "updatedAt": "2024-01-01T10:00:00"
}
```

### Transaction
```json
{
  "id": 1,
  "transactionType": "DEPOSIT",
  "amount": 1000.00,
  "description": "Initial deposit",
  "transactionDate": "2024-01-01T10:00:00",
  "user": {
    "id": 1
  },
  "status": "COMPLETED"
}
```

## Performance Testing

The system is designed to handle 30,000+ records efficiently:

- **Database Indexing**: Optimized queries with proper indexing
- **Connection Pooling**: Configured for high concurrency
- **Caching**: Ready for Redis integration
- **Pagination**: Implemented for large datasets

### Load Testing Commands
```bash
# Using Apache Bench
ab -n 1000 -c 10 http://localhost:8080/api/transactions/

# Using JMeter (recommended for complex scenarios)
# Import the provided JMeter test plan
```

## Security Features

- **JWT Authentication**: Stateless token-based auth
- **Password Encryption**: BCrypt hashing
- **Input Validation**: Comprehensive validation
- **CORS Configuration**: Cross-origin support
- **SQL Injection Protection**: Parameterized queries
- **XSS Protection**: Input sanitization

## Testing

Run the tests:
```bash
mvn test
```

## Docker Support

Build Docker image:
```bash
docker build -t transaction-api .
```

Run with Docker Compose:
```bash
docker-compose up
```

## Monitoring & Logging

- **Application Logs**: Configured logging levels
- **Database Logs**: SQL query logging enabled
- **Performance Metrics**: Ready for Actuator integration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.