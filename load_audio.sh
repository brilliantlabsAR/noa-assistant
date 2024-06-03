#!/bin/bash
# Set your SSH server details based on user's choice
SSH_USER=root
SSH_HOST=139.144.72.206
REMOTE_SCRIPT_PATH="audio_load.sh"
REMOTE_AUDIO_DIR="audio_logs/"
LOCAL_AUDIO_DIR="./audio_logs/"

# Ensure the local audio directory exists
mkdir -p $LOCAL_AUDIO_DIR

# Perform SSH command to run audio_load.sh
ssh $SSH_USER@$SSH_HOST "bash $REMOTE_SCRIPT_PATH"

# Check SSH exit code
if [ $? -ne 0 ]; then
    echo "SSH command execution failed"
    exit $?
fi

# Perform SCP transfer to copy audio files locally
scp -r $SSH_USER@$SSH_HOST:$REMOTE_AUDIO_DIR* $LOCAL_AUDIO_DIR

# Check SCP exit code
if [ $? -ne 0 ]; then
    echo "SCP transfer failed"
    exit $?
fi

echo "Audio files copied to $LOCAL_AUDIO_DIR successfully"
exit 0
