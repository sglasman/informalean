param(
    [Parameter(Mandatory)][string]$InstanceId,
    [Parameter(Mandatory)][string]$LocalPath,
    [Parameter(Mandatory)][string]$RemotePath
)

$ErrorActionPreference = "Stop"

Write-Host "Fetching instance details for $InstanceId..."
$json = vastai show instance $InstanceId --raw | Out-String
$instance = $json | ConvertFrom-Json

$ip = $instance.public_ipaddr
$port = $instance.ports.'22/tcp'[0].HostPort

Write-Host "Copying $LocalPath -> ${ip}:${port}:$RemotePath"
scp -r -P $port -o StrictHostKeyChecking=no $LocalPath "root@${ip}:${RemotePath}"
Write-Host "Done."
