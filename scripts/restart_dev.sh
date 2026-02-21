#!/bin/bash
# Restart all dev deployments (rolling restart, zero downtime)
# Usage: bash scripts/restart_dev.sh

set -e

NAMESPACE="dev"

DEPLOYMENTS=(
  "delineate-ai-app-dev"
  "delineate-ai-default-celery"
  "delineate-ai-cpu-celery"
  "delineate-ai-general-extraction-celery"
)

echo "Restarting all deployments in namespace: $NAMESPACE"
echo "=================================================="

for deploy in "${DEPLOYMENTS[@]}"; do
  echo "Restarting $deploy..."
  kubectl rollout restart deployment/"$deploy" -n "$NAMESPACE"
done

echo ""
echo "Waiting for rollouts to complete..."
echo "=================================================="

for deploy in "${DEPLOYMENTS[@]}"; do
  echo "Waiting for $deploy..."
  kubectl rollout status deployment/"$deploy" -n "$NAMESPACE" --timeout=300s
done

echo ""
echo "All deployments restarted successfully."
kubectl get pods -n "$NAMESPACE" -o wide
