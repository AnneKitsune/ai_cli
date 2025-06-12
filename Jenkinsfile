@Library('jenkins_rust') _
rustPipeline(
    enableBenchmarks: false,
    osList: ['linux', 'win', 'osx', 'freebsd'],
    rustVersion: 'stable',
    buildArgs: '--release',
    artifactPatterns: ['target/release/*']
)
