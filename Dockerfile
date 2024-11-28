# Base image for Glue PySpark
FROM amazon/aws-glue-libs:glue_libs_4.0.0_image_01

# Switch to root user for system updates and AWS CLI installation
USER root

# Install AWS CLI and dependencies
RUN yum update -y && \
    yum install -y unzip curl && \
    if [ "$(uname -m)" = "aarch64" ]; then \
        curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"; \
    else \
        curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"; \
    fi && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm -rf awscliv2.zip aws && \
    yum clean all

# Switch back to non-root user
USER glue_user

# Set working directory
WORKDIR /home/glue_user/workspace
