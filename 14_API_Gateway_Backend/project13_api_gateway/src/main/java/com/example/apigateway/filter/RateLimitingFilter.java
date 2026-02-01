package com.example.apigateway.filter;

import io.github.bucket4j.Bandwidth;
import io.github.bucket4j.Bucket;
import io.github.bucket4j.Refill;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.cloud.gateway.filter.GatewayFilter;
import org.springframework.cloud.gateway.filter.GatewayFilterChain;
import org.springframework.cloud.gateway.filter.factory.AbstractGatewayFilterFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.http.HttpStatus;
import org.springframework.http.server.reactive.ServerHttpResponse;
import org.springframework.stereotype.Component;
import org.springframework.web.server.ServerWebExchange;
import reactor.core.publisher.Mono;

import java.time.Duration;
import java.util.concurrent.TimeUnit;

@Component
public class RateLimitingFilter extends AbstractGatewayFilterFactory<RateLimitingFilter.Config> {

    private final RedisTemplate<String, String> redisTemplate;

    @Value("${rate.limit.requests}")
    private int rateLimitRequests;

    @Value("${rate.limit.duration}")
    private int rateLimitDuration;

    public RateLimitingFilter(RedisTemplate<String, String> redisTemplate) {
        super(Config.class);
        this.redisTemplate = redisTemplate;
    }

    @Override
    public GatewayFilter apply(Config config) {
        return (exchange, chain) -> {
            String clientId = getClientId(exchange);

            if (clientId == null) {
                return tooManyRequestsResponse(exchange);
            }

            Bucket bucket = getBucket(clientId);

            if (bucket.tryConsume(1)) {
                return chain.filter(exchange);
            } else {
                return tooManyRequestsResponse(exchange);
            }
        };
    }

    private String getClientId(ServerWebExchange exchange) {
        // Extract client ID from JWT token or IP address
        String authHeader = exchange.getRequest().getHeaders().getFirst("Authorization");
        if (authHeader != null && authHeader.startsWith("Bearer ")) {
            // In a real implementation, you'd decode the JWT to get user ID
            // For now, using IP as fallback
            return exchange.getRequest().getRemoteAddress().getAddress().getHostAddress();
        }
        return exchange.getRequest().getRemoteAddress().getAddress().getHostAddress();
    }

    private Bucket getBucket(String clientId) {
        String key = "rate_limit:" + clientId;
        // In a production environment, you'd store bucket state in Redis
        // For simplicity, using in-memory bucket here
        return Bucket.builder()
            .addLimit(Bandwidth.classic(rateLimitRequests,
                Refill.intervally(rateLimitRequests, Duration.ofMinutes(rateLimitDuration))))
            .build();
    }

    private Mono<Void> tooManyRequestsResponse(ServerWebExchange exchange) {
        ServerHttpResponse response = exchange.getResponse();
        response.setStatusCode(HttpStatus.TOO_MANY_REQUESTS);
        response.getHeaders().add("Retry-After", "60");
        return response.setComplete();
    }

    public static class Config {
        // Configuration properties can be added here if needed
    }
}