#!/usr/bin/env groovy

/*
    This Jenkinsfile is used to provide snapshot builds using the VITO CI system. Travis is used to provide publicly accessible test results.
    This Jenkinsfile uses the Jenkins shared library. (ssh://git@git.vito.local:7999/biggeo/jenkinslib.git)
    Information about the pythonPipeline method can be found in pythonPipeline.groovy
*/

@Library('lib')_

pythonPipeline {
  package_name = 'worldcereal-classification'
  wipeout_workspace = true
  python_version = ["3.10"]
  upload_dev_wheels = false
  pipeline_triggers = [cron('H H(0-6) * * *')]
  wheel_repo = 'python-packages-public'
  wheel_repo_dev = 'python-packages-public-snapshot'
  pep440 = true
  venv_rpm_deps = ['gcc', 'gcc-c++']
  extra_env_variables = [
    'OPENEO_AUTH_METHOD=client_credentials',
  ]
  extra_env_secrets = [
    // Secrets for Terrascope openEO backend (openeo.vito.be)
    'OPENEO_AUTH_PROVIDER_ID': 'TAP/big_data_services/openeo/openeo-cropclass-service-account provider_id',
    'OPENEO_AUTH_CLIENT_ID': 'TAP/big_data_services/openeo/openeo-cropclass-service-account client_id',
    'OPENEO_AUTH_CLIENT_SECRET': 'TAP/big_data_services/openeo/openeo-cropclass-service-account client_secret',
    // Secrets for CDSE openEO backend (openeo-staging.dataspace.copernicus.eu)
    'OPENEO_AUTH_CDSE_PROVIDER_ID': 'TAP/big_data_services/openeo/openeo-cropclass-service-account-cdse provider_id',
    'OPENEO_AUTH_CDSE_CLIENT_ID': 'TAP/big_data_services/openeo/openeo-cropclass-service-account-cdse client_id',
    'OPENEO_AUTH_CDSE_CLIENT_SECRET': 'TAP/big_data_services/openeo/openeo-cropclass-service-account-cdse client_secret',
  ]
}
