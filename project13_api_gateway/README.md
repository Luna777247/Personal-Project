# API Gateway Microservices Project

A comprehensive API Gateway implementation using Spring Cloud Gateway that provides routing, authentication, rate limiting, circuit breaking, and monitoring capabilities for microservices architecture.

## Features

- **JWT Authentication**: Secure token-based authentication for API requests
- **Rate Limiting**: Distributed rate limiting using Redis and Bucket4J
- **Circuit Breaker**: Fault tolerance with Resilience4J circuit breakers
- **Request Logging**: Comprehensive logging of all requests and responses
- **Service Discovery**: Automatic service discovery and routing
- **Distributed Tracing**: Integration with Zipkin for request tracing
- **Global Exception Handling**: Centralized error handling and responses

## Architecture

The API Gateway acts as a single entry point for all client requests, routing them to appropriate microservices while applying security, rate limiting, and monitoring policies.

### Routes Configuration

- `/api/transactions/**` → Transaction Service (localhost:8081)
- `/api/banking/**` → Banking Service (localhost:8082)
- `/auth/**` → Authentication Service (localhost:8083)

## Prerequisites

- Java 17 or higher
- Maven 3.6+
- Redis (for rate limiting and caching)
- Zipkin (for distributed tracing)

## Dependencies

The project uses the following key dependencies:

- Spring Cloud Gateway
- Spring Boot Security
- JWT (io.jsonwebtoken)
- Redis
- Bucket4J (Rate Limiting)
- Resilience4J (Circuit Breaker)
- Spring Cloud Sleuth (Distributed Tracing)

## Configuration

Key configuration properties in `application.properties`:

```properties
# Server
server.port=8080

# JWT
jwt.secret=mySecretKey
jwt.expiration=86400000

# Redis
spring.data.redis.host=localhost
spring.data.redis.port=6379

# Rate Limiting
rate.limit.requests=100
rate.limit.duration=1

# Circuit Breaker
resilience4j.circuitbreaker.instances.transactionService.failureRateThreshold=50
resilience4j.circuitbreaker.instances.transactionService.waitDurationInOpenState=10000

# Distributed Tracing
spring.zipkin.base-url=http://localhost:9411
spring.sleuth.sampler.probability=1.0
```

## Building and Running

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd project13_api_gateway
   ```

2. **Build the project**
   ```bash
   mvn clean compile
   ```

3. **Run the application**
   ```bash
   mvn spring-boot:run
   ```

4. **Run tests**
   ```bash
   mvn test
   ```

## API Endpoints

### Authentication Required Routes

All routes except `/auth/**` require a valid JWT token in the Authorization header:

```
Authorization: Bearer <jwt-token>
```

### Transaction Service Routes
- `GET /api/transactions` - Get all transactions
- `POST /api/transactions` - Create new transaction
- `GET /api/transactions/{id}` - Get transaction by ID
- `PUT /api/transactions/{id}` - Update transaction
- `DELETE /api/transactions/{id}` - Delete transaction

### Banking Service Routes
- `GET /api/banking/accounts` - Get all accounts
- `POST /api/banking/accounts` - Create new account
- `GET /api/banking/accounts/{id}` - Get account by ID
- `POST /api/banking/transfer` - Transfer money between accounts

### Authentication Routes
- `POST /auth/login` - User login
- `POST /auth/register` - User registration

## Security Features

### JWT Authentication
- Validates JWT tokens on protected routes
- Extracts user information and adds to request headers
- Supports role-based access control

### Rate Limiting
- Configurable request limits per client
- Redis-backed distributed rate limiting
- Returns 429 Too Many Requests when limit exceeded

### Circuit Breaker
- Protects against cascading failures
- Automatic failover to fallback endpoints
- Configurable failure thresholds and recovery times

## Monitoring and Logging

### Request Logging
- Logs all incoming requests with client IP, method, and path
- Logs response status and processing time
- Logs errors with detailed stack traces

### Distributed Tracing
- Integration with Zipkin for request tracing
- Trace ID propagation across services
- Performance monitoring and bottleneck identification

## Error Handling

The gateway provides centralized error handling for:

- Authentication failures (401 Unauthorized)
- Rate limit exceeded (429 Too Many Requests)
- Service unavailable (503 Service Unavailable)
- Internal server errors (500 Internal Server Error)

## Development

### Project Structure

```
src/
├── main/
│   ├── java/
│   │   └── com/example/apigateway/
│   │       ├── ApiGatewayApplication.java
│   │       ├── config/
│   │       │   └── GatewayConfig.java
│   │       ├── filter/
│   │       │   ├── LoggingFilter.java
│   │       │   └── RateLimitingFilter.java
│   │       ├── security/
│   │       │   └── JwtAuthenticationFilter.java
│   │       └── exception/
│   │           └── GlobalExceptionHandler.java
│   └── resources/
│       └── application.properties
└── test/
    └── java/
        └── com/example/apigateway/
            └── ApiGatewayApplicationTests.java
```

### Adding New Routes

To add a new service route, update the `GatewayConfig.java`:

```java
.route("new-service", r -> r.path("/api/new-service/**")
    .filters(f -> f
        .filter(loggingFilter.apply(new LoggingFilter.Config()))
        .filter(rateLimitingFilter.apply(new RateLimitingFilter.Config()))
        .filter(jwtAuthenticationFilter)
        .circuitBreaker(c -> c.setName("newService")
            .setFallbackUri("forward:/fallback/new-service"))
    )
    .uri("http://localhost:8084")
)
```

## Testing

Run the included unit tests:

```bash
mvn test
```

For integration testing, ensure all dependent services are running and test through the gateway endpoints.

## Deployment

### Docker

```dockerfile
FROM openjdk:17-jdk-alpine
COPY target/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java","-jar","/app.jar"]
```

### Docker Compose

Example docker-compose.yml for the complete microservices stack:

```yaml
version: '3.8'
services:
  api-gateway:
    build: .
    ports:
      - "8080:8080"
    depends_on:
      - redis
      - zipkin

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"

  zipkin:
    image: openzipkin/zipkin
    ports:
      - "9411:9411"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.