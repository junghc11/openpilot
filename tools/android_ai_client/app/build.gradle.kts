plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

val qnnEpIncluded = providers.gradleProperty("carrotQnnEnabled")
  .map { it.toBooleanStrict() }
  .getOrElse(true)

android {
  namespace = "ai.carrotpilot.external"
  compileSdk = 35

  defaultConfig {
    applicationId = "ai.carrotpilot.external"
    minSdk = 29
    targetSdk = 35
    versionCode = 10
    versionName = "0.8.1"
    buildConfigField("boolean", "QNN_EP_INCLUDED", qnnEpIncluded.toString())
    ndk {
      abiFilters += "arm64-v8a"
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
  if (qnnEpIncluded) {
    implementation("com.microsoft.onnxruntime:onnxruntime-android-qnn:1.24.3")
  } else {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")
  }
}
