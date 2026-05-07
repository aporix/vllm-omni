#!/bin/bash

set -ex

# Pull per-arch images from ECR (built earlier in the pipeline) and re-publish
# them to Docker Hub under a nightly tag, then create and push a multi-arch
# manifest pointing at both per-arch tags.

ORIG_TAG_NAME="$BUILDKITE_COMMIT"
TAG_NAME="nightly"
DOCKERHUB_REPO="vllm/vllm-omni"
ECR_REPO="public.ecr.aws/q9t5s3a7/vllm-omni-release-repo"

echo "Pushing original tag $ORIG_TAG_NAME to new nightly tag name: $TAG_NAME"

# pull original arch-dependent images from AWS ECR Public
aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7
docker pull "$ECR_REPO:$ORIG_TAG_NAME-x86_64"
docker pull "$ECR_REPO:$ORIG_TAG_NAME-aarch64"
# tag arch-dependent images
docker tag "$ECR_REPO:$ORIG_TAG_NAME-x86_64" "$DOCKERHUB_REPO:$TAG_NAME-x86_64"
docker tag "$ECR_REPO:$ORIG_TAG_NAME-aarch64" "$DOCKERHUB_REPO:$TAG_NAME-aarch64"
# push arch-dependent images to DockerHub
docker push "$DOCKERHUB_REPO:$TAG_NAME-x86_64"
docker push "$DOCKERHUB_REPO:$TAG_NAME-aarch64"
# push arch-independent manifest to DockerHub
docker manifest create "$DOCKERHUB_REPO:$TAG_NAME" "$DOCKERHUB_REPO:$TAG_NAME-x86_64" "$DOCKERHUB_REPO:$TAG_NAME-aarch64" --amend
docker manifest create "$DOCKERHUB_REPO:$TAG_NAME-$BUILDKITE_COMMIT" "$DOCKERHUB_REPO:$TAG_NAME-x86_64" "$DOCKERHUB_REPO:$TAG_NAME-aarch64" --amend
docker manifest push "$DOCKERHUB_REPO:$TAG_NAME"
docker manifest push "$DOCKERHUB_REPO:$TAG_NAME-$BUILDKITE_COMMIT"
