# Server Configuration Changes Documentation

## Overview
This document provides a detailed account of the changes made to the server to manage RAM and swap usage efficiently, particularly concerning the finance team's modeling program which has shown to consume significant system resources.

## Changes Made

### 1. Cgroup Configuration
We have implemented control groups (cgroups) to manage and limit the RAM usage of specific processes, notably the finance modeling script.

**Details**:
- **Cgroup Path**: `/sys/fs/cgroup/tradebot`
- **Memory Limit**: Set to 40 GB to prevent the process from consuming all available RAM, which could lead to system instability.

### 2. Swap Configuration
To manage the swap usage, which was previously maxed out, we've implemented additional swap space.

**Details**:
- **Additional Swap File**: Created a 2GB swap file to increase the available swap space, thus providing a larger buffer for in-memory operations when physical RAM is exhausted.
- **Swap File Location**: `/swapfile`
- **Size**: 2GB
- **Permissions**: Set to 600 to ensure that the swap file is only accessible by the root user.

### 3. Swappiness Adjustment
We adjusted the system's swappiness parameter to make it less aggressive in swapping out runtime memory.

**Details**:
- **Swappiness Value**: Changed from the default (likely 60) to 10, to decrease the likelihood of swapping, favoring keeping more data in physical RAM as long as possible.

## Commands Used
```bash
# Cgroup creation and process assignment
sudo cgcreate -g memory:/tradebot
echo '42949672960' | sudo tee /sys/fs/cgroup/tradebot/memory.max
echo 'PID' > /sys/fs/cgroup/tradebot/cgroup.procs

# Swap file creation and activation
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab

# Swappiness adjustment
sudo sysctl vm.swappiness=10
echo 'vm.swappiness = 10' | sudo tee -a /etc/sysctl.conf
