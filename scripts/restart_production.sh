#!/bin/bash
# Restart all production deployments (rolling restart, zero downtime)
# Usage: bash scripts/restart_production.sh

set -e

NAMESPACE="production"

DEPLOYMENTS=(
  "delineate-ai-app-production"
  "delineate-ai-default-celery"
  "delineate-ai-cpu-celery"
  "delineate-ai-general-extraction-celery"
)

echo "⚠ WARNING: You are about to restart PRODUCTION deployments."
read -p "Are you sure? (y/N): " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
  echo "Aborted."
  exit 0
fi

echo ""
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
