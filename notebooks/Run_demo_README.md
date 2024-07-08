In order to run these demo notebooks smoothly, the WorldCereal consortium has set up a pre-configured Python environment for you.

To access this environment and run the different notebooks, please follow the links below.

However, BEFORE being able to make use of this service, you will need to register for the EGI notebooks service,
through the following link:

https://aai.egi.eu/registry/co_petitions/start/coef:111


Once this registration process has been completed, follow the links below to start the actual demo's.


********************************************************

1) Demonstration on how to generate default WorldCereal products:

xxx


********************************************************

2) Demonstration on how to generate custom WorldCereal products:

xxx



### IMPORTANT INFORMATION TO ENSURE DISPLAY OF RESULTS WORKS PROPERLY IN THE NOTEBOOK

if working on binder, set localtileserver client prefix
import os
os.environ['LOCALTILESERVER_CLIENT_PREFIX'] = "proxy/{{port}}"

if working on terrascope virtual machine, ensure that you forward the port of the localtileserver
1) in the add_raster function, add the following argument: port=LOCALTILESERVER_PORT
2) ensure that whichever number you defined as the LOCALTILESERVER_PORT, this port is forwarded to your local machine
e.g. Port 7777, Forwarded address: localhost:7778