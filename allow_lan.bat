@echo off
rem Run once as Administrator: lets other devices on the private network reach port 5000.
netsh advfirewall firewall add rule name="LEGO guidance (LAN)" dir=in action=allow protocol=TCP localport=5000 profile=private
pause
