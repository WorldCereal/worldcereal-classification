#!/usr/bin/env groovy

/* Jenkinsfile for snapshot building with VITO CI system. */

@Library('lib')_

pythonPipeline {
  package_name = 'worldcereal-classification'
  test_module_name = 'worldcereal'
  wipeout_workspace = true
  python_version = ["3.10"]
  extras_require = 'dev'
  upload_dev_wheels = false
  pipeline_triggers = [cron('H H(0-6) * * *')]
  pep440 = true
  extra_env_variables = [
    "OPENEO_AUTH_METHOD=client_credentials",
    "OPENEO_OIDC_DEVICE_CODE_MAX_POLL_TIME=5",
    "OPENEO_AUTH_PROVIDER_ID_TERRASCOPE=terrascope",
    "OPENEO_AUTH_CLIENT_ID_TERRASCOPE=openeo-worldcereal-service-account",
    "OPENEO_AUTH_PROVIDER_ID_CDSE=CDSE",
    "OPENEO_AUTH_CLIENT_ID_CDSE=openeo-worldcereal-service-account",
  ]
  extra_env_secrets = [
    'OPENEO_AUTH_CLIENT_SECRET_TERRASCOPE': 'TAP/big_data_services/devops/terraform/keycloak_mgmt/oidc_clients_prod openeo-worldcereal-service-account',
    'OPENEO_AUTH_CLIENT_SECRET_CDSE': 'TAP/big_data_services/openeo/cdse-service-accounts/openeo-worldcereal-service-account client_secret',
  ]
}
