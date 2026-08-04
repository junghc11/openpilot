plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

val qnnEpIncluded = providers.gradleProperty("carrotQnnEnabled")
  .map { it.toBooleanStrict() }
  .getOrElse(true)
val targetAbi = providers.gradleProperty("carrotTargetAbi").getOrElse("arm64-v8a")

android {
  namespace = "ai.carrotpilot.external"
  compileSdk = 35

  defaultConfig {
    applicationId = "ai.carrotpilot.external"
    minSdk = 29
    targetSdk = 35
    versionCode = 19
    versionName = "0.14.0"
    buildConfigField("boolean", "QNN_EP_INCLUDED", qnnEpIncluded.toString())
    ndk {
      abiFilters += targetAbi
    }
  }

  buildFeatures {
    buildConfig = true
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
  }

  kotlinOptions {
    jvmTarget = "17"
  }

  lint {
    // This is a dedicated Qualcomm/ARM phone client, not a ChromeOS application.
    disable += "ChromeOsAbiSupport"
  }

  packaging {
    jniLibs.useLegacyPackaging = true
  }
}

dependencies {
  implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")
  testImplementation("junit:junit:4.13.2")
  if (qnnEpIncluded) {
    runtimeOnly("com.qualcomm.qti:onnxruntime-android-qnn:2.4.0")
    runtimeOnly("com.qualcomm.qti:qnn-runtime:2.48.0")
  }
}
