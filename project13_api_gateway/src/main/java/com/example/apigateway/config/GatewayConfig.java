package com.example.apigateway.config;

import com.example.apigateway.filter.LoggingFilter;
import com.example.apigateway.filter.RateLimitingFilter;
import com.example.apigateway.security.JwtAuthenticationFilter;
import org.springframework.cloud.gateway.route.RouteLocator;
import org.springframework.cloud.gateway.route.builder.RouteLocatorBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class GatewayConfig {

    private final JwtAuthenticationFilter jwtAuthenticationFilter;
    private final RateLimitingFilter rateLimitingFilter;
    private final LoggingFilter loggingFilter;

    public GatewayConfig(JwtAuthenticationFilter jwtAuthenticationFilter,
                        RateLimitingFilter rateLimitingFilter,
                        LoggingFilter loggingFilter) {
        this.jwtAuthenticationFilter = jwtAuthenticationFilter;
        this.rateLimitingFilter = rateLimitingFilter;
        this.loggingFilter = loggingFilter;
    }

    @Bean
    public RouteLocator customRouteLocator(RouteLocatorBuilder builder) {
        return builder.routes()
            .route("transaction-service", r -> r.path("/api/transactions/**")
                .filters(f -> f
                    .filter(loggingFilter.apply(new LoggingFilter.Config()))
                    .filter(rateLimitingFilter.apply(new RateLimitingFilter.Config()))
                    .filter(jwtAuthenticationFilter)
                    .circuitBreaker(c -> c.setName("transactionService")
                        .setFallbackUri("forward:/fallback/transaction"))
                )
                .uri("http://localhost:8081")
            )
            .route("banking-service", r -> r.path("/api/banking/**")
                .filters(f -> f
                    .filter(loggingFilter.apply(new LoggingFilter.Config()))
                    .filter(rateLimitingFilter.apply(new RateLimitingFilter.Config()))
                    .filter(jwtAuthenticationFilter)
                    .circuitBreaker(c -> c.setName("bankingService")
                        .setFallbackUri("forward:/fallback/banking"))
                )
                .uri("http://localhost:8082")
            )
            .route("auth-service", r -> r.path("/auth/**")
                .filters(f -> f
                    .filter(loggingFilter.apply(new LoggingFilter.Config()))
                )
                .uri("http://localhost:8083")
            )
            .build();
    }
}