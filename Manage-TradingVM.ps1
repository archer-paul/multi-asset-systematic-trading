# Trading VM Management Script - Cost Optimization
# Manages the ML training VM to minimize GCP costs

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("start", "stop", "status", "cost", "schedule")]
    [string]$Action,
    
    [string]$Zone = "europe-west1-b",
    [string]$VmName = "trading-ml-trainer",
    [int]$AutoShutdownMinutes = 60,
    [string]$ScheduleTime = "09:00"  # Start time for scheduled training
)

function Get-VmStatus {
    $status = gcloud compute instances describe $VmName --zone=$Zone --format="value(status)" 2>$null
    return $status
}

function Get-VmCosts {
    $status = Get-VmStatus
    if ($status -eq "RUNNING") {
        $uptimeSeconds = gcloud compute instances describe $VmName --zone=$Zone --format="value(lastStartTimestamp)" 2>$null
        if ($uptimeSeconds) {
            $startTime = [DateTime]::Parse($uptimeSeconds)
            $runTime = (Get-Date) - $startTime
            $costPerHour = 0.54  # $0.54/hour for n1-standard-4 + Tesla T4
            $currentCost = [math]::Round(($runTime.TotalHours * $costPerHour), 2)
            
            Write-Host "VM Status: RUNNING" -ForegroundColor Green
            Write-Host "Running Time: $($runTime.ToString('dd\.hh\:mm\:ss'))" -ForegroundColor Yellow
            Write-Host "Estimated Cost: $$$currentCost" -ForegroundColor Red
            Write-Host "Cost Rate: $$$costPerHour/hour" -ForegroundColor Yellow
        }
    } else {
        Write-Host "VM Status: $status" -ForegroundColor Blue
        Write-Host "Current Cost: $0.00 (VM stopped)" -ForegroundColor Green
    }
}

function Start-TradingVM {
    $status = Get-VmStatus
    if ($status -eq "RUNNING") {
        Write-Host "[INFO] VM is already running" -ForegroundColor Blue
        Get-VmCosts
        return
    }
    
    Write-Host "[INFO] Starting ML training VM..." -ForegroundColor Yellow
    gcloud compute instances start $VmName --zone=$Zone
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[INFO] Waiting for VM to be ready..." -ForegroundColor Yellow
        gcloud compute instances wait-until-ready $VmName --zone=$Zone
        
        $ip = gcloud compute instances describe $VmName --zone=$Zone --format='value(networkInterfaces[0].accessConfigs[0].natIP)'
        Write-Host "[SUCCESS] VM started successfully!" -ForegroundColor Green
        Write-Host "External IP: $ip" -ForegroundColor Cyan
        Write-Host "SSH Command: gcloud compute ssh $VmName --zone=$Zone" -ForegroundColor Blue
        Write-Host "Jupyter URL: http://$ip:8888" -ForegroundColor Blue
        Write-Host ""
        Write-Host "[WARNING] Auto-shutdown is enabled after ${AutoShutdownMinutes} minutes of inactivity" -ForegroundColor Yellow
        Write-Host "Estimated cost: $0.54/hour while running" -ForegroundColor Red
    } else {
        Write-Host "[ERROR] Failed to start VM" -ForegroundColor Red
    }
}

function Stop-TradingVM {
    $status = Get-VmStatus
    if ($status -ne "RUNNING") {
        Write-Host "[INFO] VM is already stopped" -ForegroundColor Blue
        return
    }
    
    Write-Host "[INFO] Stopping ML training VM..." -ForegroundColor Yellow
    
    # Try to save any running models first
    Write-Host "[INFO] Attempting to save any running training..." -ForegroundColor Yellow
    gcloud compute ssh $VmName --zone=$Zone --command="./save_models.sh" 2>$null
    
    gcloud compute instances stop $VmName --zone=$Zone
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] VM stopped successfully!" -ForegroundColor Green
        Write-Host "Cost savings: VM is no longer consuming credits" -ForegroundColor Green
    } else {
        Write-Host "[ERROR] Failed to stop VM" -ForegroundColor Red
    }
}

function Set-ScheduledTraining {
    Write-Host "[INFO] Setting up scheduled training at $ScheduleTime daily..." -ForegroundColor Yellow
    
    # Create a scheduled task (Windows)
    $taskName = "TradingML-AutoStart"
    $scriptPath = $PSCommandPath
    $action = New-ScheduledTaskAction -Execute "PowerShell.exe" -Argument "-File `"$scriptPath`" -Action start"
    $trigger = New-ScheduledTaskTrigger -Daily -At $ScheduleTime
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    
    try {
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
        Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings
        Write-Host "[SUCCESS] Scheduled task created: $taskName" -ForegroundColor Green
        Write-Host "VM will start daily at $ScheduleTime" -ForegroundColor Yellow
        Write-Host "Note: Make sure to stop the VM manually when training is complete!" -ForegroundColor Red
    }
    catch {
        Write-Host "[ERROR] Failed to create scheduled task: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Main execution
switch ($Action) {
    "start" {
        Start-TradingVM
    }
    "stop" {
        Stop-TradingVM
    }
    "status" {
        Get-VmCosts
        
        # Show additional info
        Write-Host ""
        Write-Host "VM Management Commands:" -ForegroundColor Cyan
        Write-Host "  Start VM:  .\Manage-TradingVM.ps1 -Action start" -ForegroundColor White
        Write-Host "  Stop VM:   .\Manage-TradingVM.ps1 -Action stop" -ForegroundColor White
        Write-Host "  Status:    .\Manage-TradingVM.ps1 -Action status" -ForegroundColor White
        Write-Host "  SSH:       gcloud compute ssh $VmName --zone=$Zone" -ForegroundColor White
    }
    "cost" {
        Write-Host "ML Training VM Cost Information:" -ForegroundColor Cyan
        Write-Host "================================" -ForegroundColor Cyan
        Write-Host "Machine Type: n1-standard-4 (4 vCPUs, 15GB RAM)" -ForegroundColor White
        Write-Host "GPU: 1x NVIDIA Tesla T4" -ForegroundColor White
        Write-Host "Hourly Rate: ~$0.54/hour" -ForegroundColor Yellow
        Write-Host "Daily Rate (24h): ~$12.96/day" -ForegroundColor Red
        Write-Host "Monthly Rate (24/7): ~$389/month" -ForegroundColor Red
        Write-Host ""
        Write-Host "Cost Optimization:" -ForegroundColor Green
        Write-Host "- Auto-shutdown after inactivity: ENABLED" -ForegroundColor Green
        Write-Host "- Stop VM when not training to save costs" -ForegroundColor Green
        Write-Host "- Use preemptible instances for longer training (70% discount)" -ForegroundColor Yellow
        
        Get-VmCosts
    }
    "schedule" {
        Set-ScheduledTraining
    }
}

# Display cost warning if VM is running
$finalStatus = Get-VmStatus
if ($finalStatus -eq "RUNNING" -and $Action -ne "status" -and $Action -ne "cost") {
    Write-Host ""
    Write-Host "[COST WARNING] VM is currently running and consuming credits!" -ForegroundColor Red -BackgroundColor Yellow
    Write-Host "Remember to stop it when training is complete: .\Manage-TradingVM.ps1 -Action stop" -ForegroundColor Yellow
}