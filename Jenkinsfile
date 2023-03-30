#!/usr/bin/env groovy

/*
    This Jenkinsfile is used to provide snapshot builds using the VITO CI system. Travis is used to provide publicly accessible test results.
    This Jenkinsfile uses the Jenkins shared library. (ssh://git@git.vito.local:7999/biggeo/jenkinslib.git)
    Information about the pythonPipeline method can be found in pythonPipeline.groovy
*/

@Library('lib')_

pythonPipeline {
  package_name = 'worldcereal'
  wipeout_workspace = true
  python_version = ["3.8"]
  upload_dev_wheels = true
  hadoop = true
  extra_container_volumes = [
    '/data/worldcereal:/data/worldcereal:ro',
    '/data/MTDA/AgERA5:/data/MTDA/AgERA5:ro']
  pipeline_triggers = [cron('H 1 * * *')]
  wheel_repo = 'python-packages-public'
  wheel_repo_dev = 'python-packages-public-snapshot'
  system_site_packages = 'nope'
  pep440 = true
  venv_rpm_deps = ['gcc', 'gcc-c++']
}
