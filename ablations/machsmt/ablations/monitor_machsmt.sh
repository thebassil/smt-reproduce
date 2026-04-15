#!/bin/bash
# Monitor MachSMT ablation job 1267046
# Checks every 5 minutes, reports on completion/failure

JOB_ID=1269428
LOG_DIR="/dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/logs"

while true; do
    # Get job states
    STATES=$(sacct -j $JOB_ID --format=JobID%15,State%12,ExitCode -P --noheader 2>/dev/null | grep -v '\.batch')
    
    if [ -z "$STATES" ]; then
        echo "[$(date '+%H:%M')] Job $JOB_ID not found in sacct yet"
        sleep 300
        continue
    fi
    
    PENDING=$(echo "$STATES" | grep -c "PENDING")
    RUNNING=$(echo "$STATES" | grep -c "RUNNING")
    COMPLETED=$(echo "$STATES" | grep -c "COMPLETED")
    FAILED=$(echo "$STATES" | grep -c "FAILED\|TIMEOUT\|OUT_OF_ME\|CANCELLED")
    TOTAL=$(echo "$STATES" | wc -l)
    
    echo "[$(date '+%H:%M')] P:$PENDING R:$RUNNING C:$COMPLETED F:$FAILED / $TOTAL"
    
    # If any failed, dump diagnostics and exit
    if [ "$FAILED" -gt 0 ]; then
        echo ""
        echo "=== FAILURES DETECTED ==="
        echo "$STATES" | grep -E "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED"
        echo ""
        # Get the failed task IDs
        for TASK_ID in $(echo "$STATES" | grep -E "FAILED|TIMEOUT|OUT_OF_ME|CANCELLED" | head -3 | cut -d'|' -f1 | sed 's/.*_//'); do
            echo "=== STDERR task $TASK_ID (last 20 lines) ==="
            tail -20 "$LOG_DIR/machsmt-abl_${JOB_ID}_${TASK_ID}.err" 2>/dev/null
            echo ""
            echo "=== STDOUT task $TASK_ID (last 10 lines) ==="
            tail -10 "$LOG_DIR/machsmt-abl_${JOB_ID}_${TASK_ID}.out" 2>/dev/null
            echo ""
        done
        exit 1
    fi
    
    # If all completed successfully
    if [ "$COMPLETED" -eq "$TOTAL" ] && [ "$TOTAL" -gt 0 ]; then
        echo ""
        echo "=== ALL $TOTAL TASKS COMPLETED SUCCESSFULLY ==="
        echo ""
        # Show result summaries
        for f in /dcs/23/u5573765/cs351/smt-ablation-machsmt/ablations/results/*_summary.csv; do
            if [ -f "$f" ]; then
                NAME=$(basename "$f")
                ROWS=$(wc -l < "$f")
                ZERO_SOLVE=$(grep -c ',0\.0,0\.0,' "$f" 2>/dev/null || echo 0)
                echo "$NAME: $((ROWS-1)) rows, $ZERO_SOLVE with 0% solve"
            fi
        done
        exit 0
    fi
    
    # Still in progress — check running tasks for partial output
    if [ "$RUNNING" -gt 0 ]; then
        for TASK_ID in $(echo "$STATES" | grep "RUNNING" | head -1 | cut -d'|' -f1 | sed 's/.*_//'); do
            LAST_LINE=$(tail -1 "$LOG_DIR/machsmt-abl_${JOB_ID}_${TASK_ID}.out" 2>/dev/null)
            echo "  Task $TASK_ID progress: $LAST_LINE"
        done
    fi
    
    sleep 300
done
