#!/usr/bin/env groovy

/* Jenkinsfile for snapshot building with VITO CI system. */

@Library('lib')_

pythonPipeline {
  package_name = 'worldcereal-classification'
  wipeout_workspace = true
  python_version = ["3.10"]
  extras_require = 'dev'
  upload_dev_wheels = false
  pipeline_triggers = [cron('H H(0-6) * * *')]
  pep440 = true
}
