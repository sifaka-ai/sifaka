# Monitoring Integration Examples

This directory contains examples of integrating Sifaka with various monitoring and observability platforms.

## Available Examples

### 1. OpenTelemetry Integration (`opentelemetry_integration.py`)

Comprehensive observability with distributed tracing and metrics:

```python
from examples.monitoring.opentelemetry_integration import setup_opentelemetry

# Set up OpenTelemetry
otel_middleware = setup_opentelemetry(
    service_name="my-sifaka-app",
    otlp_endpoint="localhost:4317"
)

# Use with Sifaka
result = improve_sync(
    text,
    middleware=[otel_middleware]
)
```

**Features:**
- Distributed tracing with automatic span creation
- Metrics collection (counters, histograms, gauges)
- Context propagation for microservices
- Support for multiple exporters (OTLP, Jaeger, Zipkin)

**Running the Collector:**
```bash
# Using Docker
docker run -p 4317:4317 -p 55679:55679 otel/opentelemetry-collector:latest

# View traces in Jaeger
docker run -p 16686:16686 jaegertracing/all-in-one:latest
```

### 2. Prometheus Integration (`prometheus_integration.py`)

Metrics collection in Prometheus format:

```python
from examples.monitoring.prometheus_integration import setup_prometheus

# Set up Prometheus metrics endpoint
prom_middleware = setup_prometheus(port=8000)

# Metrics available at http://localhost:8000/metrics
```

**Metrics Exposed:**
- `sifaka_improvements_total` - Counter of improvement operations
- `sifaka_improvement_duration_seconds` - Histogram of operation duration
- `sifaka_improvement_iterations` - Histogram of iterations per improvement
- `sifaka_tokens_total` - Counter of tokens consumed
- `sifaka_errors_total` - Counter of errors by type

**Example Prometheus Queries:**
```promql
# Request rate
rate(sifaka_improvements_total[5m])

# P95 latency
histogram_quantile(0.95, rate(sifaka_improvement_duration_seconds_bucket[5m]))

# Error rate
rate(sifaka_errors_total[5m]) / rate(sifaka_improvements_total[5m])
```

### 3. DataDog Integration (`datadog_integration.py`)

APM, metrics, and logs with DataDog:

```python
from examples.monitoring.datadog_integration import setup_datadog

# Set up DataDog
dd_middleware = setup_datadog(
    service_name="sifaka-prod",
    environment="production",
    enable_profiling=True
)
```

**Features:**
- Automatic APM instrumentation
- StatsD metrics with tags
- Event tracking for errors
- Continuous profiling support
- Custom business metrics

**DataDog Agent Setup:**
```bash
# Install DataDog agent
DD_API_KEY=your-api-key bash -c "$(curl -L https://s3.amazonaws.com/dd-agent/scripts/install_script.sh)"

# Enable APM
echo "apm_enabled: true" >> /etc/datadog-agent/datadog.yaml
```

### 4. Custom Logging Configuration (`logging_configuration.py`)

Flexible logging for different environments:

```python
from examples.monitoring.logging_configuration import setup_logging, LoggingMiddleware

# Set up logging
logger = setup_logging("production")  # or "development", "elk", "cloudwatch"

# Create middleware
log_middleware = LoggingMiddleware(logger)
```

**Available Configurations:**
- **Development** - Colored console output with debug info
- **Production** - Structured JSON logs with rotation
- **ELK** - Logstash integration for Elasticsearch
- **CloudWatch** - AWS CloudWatch Logs integration

## Middleware Pattern

All monitoring integrations use Sifaka's middleware pattern:

```python
from sifaka.core.middleware import BaseMiddleware

class MyMonitoringMiddleware(BaseMiddleware):
    async def pre_improve(self, text: str, **kwargs):
        # Called before improvement starts
        return {"start_time": time.time()}

    async def post_improve(self, result, context, error=None):
        # Called after improvement completes
        duration = time.time() - context["start_time"]
        # Send metrics
```

## Best Practices

### 1. Metric Design

**Choose Meaningful Metrics:**
- **Business Metrics**: improvement success rate, user satisfaction, cost per improvement
- **Technical Metrics**: latency, throughput, error rate, resource usage
- **Quality Metrics**: text improvement ratio, iteration count, validation pass rate

**Metric Naming Conventions:**
```python
# Good: Clear, hierarchical, units included
sifaka.improvement.duration_seconds
sifaka.tokens.used_total
sifaka.errors.by_type

# Bad: Unclear, no units, flat structure
duration
token_count
errors
```

**Cardinality Management:**
```python
# Bad: High cardinality (too many unique values)
metrics.increment("user.action", tags=[f"user_id:{user_id}"])  # Millions of users

# Good: Bounded cardinality
metrics.increment("user.action", tags=[
    f"user_segment:{get_segment(user_id)}",  # e.g., "free", "pro", "enterprise"
    f"action_type:{action_type}"
])
```

### 2. Environment-Based Configuration

```python
import os

# Choose monitoring based on environment
if os.getenv("ENV") == "production":
    middleware = setup_datadog(environment="production")
elif os.getenv("ENV") == "staging":
    middleware = setup_prometheus()
else:
    middleware = LoggingMiddleware()
```

### 2. Multiple Middleware

```python
# Combine multiple monitoring solutions
result = improve_sync(
    text,
    middleware=[
        prometheus_middleware,
        logging_middleware,
        custom_metrics_middleware
    ]
)
```

### 3. Error Handling

```python
# Monitoring shouldn't break your app
try:
    middleware = setup_opentelemetry()
except Exception as e:
    logger.warning(f"Failed to setup monitoring: {e}")
    middleware = None

result = improve_sync(
    text,
    middleware=[middleware] if middleware else []
)
```

### 4. Performance Considerations

- Use sampling for high-volume applications
- Batch metrics to reduce overhead
- Use async exporters when possible
- Set appropriate buffer sizes

### 5. Security

- Never log sensitive data (API keys, PII)
- Use TLS for metric endpoints
- Implement authentication for metrics endpoints
- Rotate logs with sensitive information

### 6. Alerting Strategy

**Alert on Symptoms, Not Causes:**
```yaml
# Good: User-facing impact
- alert: HighErrorRate
  expr: rate(sifaka_errors_total[5m]) > 0.05
  annotations:
    summary: "High error rate affecting users"

# Bad: Internal metric
- alert: HighMemoryUsage
  expr: process_resident_memory_bytes > 1e9
```

**Alert Fatigue Prevention:**
- Set appropriate thresholds based on historical data
- Use alert grouping and routing
- Implement alert suppression during maintenance
- Regular alert review and tuning

### 7. Cost Management

**Sampling Strategies:**
```python
# Head-based sampling (decide upfront)
if random.random() < 0.1:  # 10% sampling
    with tracer.start_span("operation"):
        # Traced operation

# Tail-based sampling (decide after completion)
def should_sample(span):
    return span.duration > 5.0 or span.has_error
```

**Data Retention:**
- High-resolution data: 7 days
- 5-minute aggregates: 30 days
- Hourly aggregates: 1 year
- Daily aggregates: 2 years

### 8. Performance Monitoring

**Key Metrics to Track:**
```python
# Latency percentiles
histogram_quantile(0.50, ...)  # p50 (median)
histogram_quantile(0.95, ...)  # p95
histogram_quantile(0.99, ...)  # p99

# Throughput
rate(sifaka_improvements_total[5m])

# Error budget
1 - (rate(sifaka_errors_total[30d]) / rate(sifaka_improvements_total[30d]))
```

**SLI/SLO Definition:**
```yaml
slis:
  - name: availability
    query: 1 - (rate(errors[5m]) / rate(requests[5m]))

  - name: latency
    query: histogram_quantile(0.95, improvement_duration_bucket) < 10

slos:
  - sli: availability
    objective: 0.999  # 99.9%

  - sli: latency
    objective: 0.95   # 95% of requests under 10s
```

### 9. Debugging with Monitoring

**Correlation IDs:**
```python
import uuid

class CorrelationMiddleware(BaseMiddleware):
    async def pre_improve(self, text: str, **kwargs):
        correlation_id = str(uuid.uuid4())
        logger.info("Starting improvement", extra={"correlation_id": correlation_id})
        return {"correlation_id": correlation_id}
```

**Distributed Tracing:**
```python
# Trace context propagation
headers = {
    "traceparent": f"00-{trace_id}-{span_id}-01",
    "tracestate": "sifaka=xyz"
}
```

### 10. Monitoring in Development

**Local Monitoring Stack:**
```yaml
# docker-compose.yml
version: '3'
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"

  jaeger:
    image: jaegertracing/all-in-one
    ports:
      - "16686:16686"
```

**Synthetic Monitoring:**
```python
# Generate test traffic for monitoring
async def synthetic_monitor():
    while True:
        try:
            result = await improve_async(
                "Test text for monitoring",
                metadata={"synthetic": True}
            )
            metrics.increment("synthetic.success")
        except Exception as e:
            metrics.increment("synthetic.failure")
            logger.error(f"Synthetic check failed: {e}")

        await asyncio.sleep(60)  # Every minute
```

## Monitoring Dashboards

### OpenTelemetry + Grafana

1. Import dashboard template: `dashboards/sifaka-otel.json`
2. Key panels:
   - Request rate by critic
   - Latency percentiles
   - Token usage over time
   - Error rate and types

### Prometheus + Grafana

1. Import dashboard template: `dashboards/sifaka-prometheus.json`
2. Alerts configuration: `alerts/sifaka-alerts.yml`

### DataDog

1. Use the Sifaka Dashboard template
2. Set up monitors for:
   - High error rate
   - Slow improvements
   - Token usage spikes

## Troubleshooting

### No Metrics Appearing

1. Check middleware is properly initialized
2. Verify monitoring backend is running
3. Check network connectivity
4. Review logs for errors

### High Overhead

1. Reduce metric cardinality (fewer tags)
2. Increase batching intervals
3. Use sampling for traces
4. Disable debug logging

### Missing Traces

1. Ensure trace context propagation
2. Check sampling configuration
3. Verify span relationships
4. Review trace exporter logs

## Production Checklist

- [ ] Choose appropriate monitoring solution(s)
- [ ] Configure retention policies
- [ ] Set up dashboards and alerts
- [ ] Test monitoring in staging
- [ ] Document runbooks for alerts
- [ ] Configure log rotation
- [ ] Set up backup monitoring
- [ ] Test monitoring failover
- [ ] Review security settings
- [ ] Load test monitoring overhead

## Additional Resources

- [OpenTelemetry Best Practices](https://opentelemetry.io/docs/reference/specification/overview/)
- [Prometheus Metric Naming](https://prometheus.io/docs/practices/naming/)
- [DataDog APM Guide](https://docs.datadoghq.com/tracing/)
- [ELK Stack Tutorial](https://www.elastic.co/guide/index.html)
- [CloudWatch Insights](https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/AnalyzingLogData.html)
