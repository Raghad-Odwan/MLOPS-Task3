# Monitoring Notes

## What is tracked

- **Service metrics** (`/metrics` route): request count, error count and rate,
  average latency, and the distribution of predictions (late vs on-time).
- **Prediction log** (`logs/prediction_log.csv`): every single prediction is
  appended with its input order details, the prediction, probability, model
  version, and latency. This lets us later join predictions against the real
  delivery outcome once it becomes known, to measure real-world accuracy.

## What we would alert on

- **Error rate above 5%** over a rolling window: suggests the service is
  receiving malformed data or the model/pipeline is broken.
- **Average latency above 1000ms**: suggests a performance problem (e.g. a
  slow database, an overloaded container) that would affect user experience.
- **Predicted late ratio drifting far from the training baseline (~8%)**:
  if the fraction of "late" predictions suddenly jumps to, say, 25% or drops
  to 0% over a large number of requests, that is a sign of data drift (the
  incoming orders look statistically different from what the model was
  trained on) and the model should be reviewed/retrained.
