plugins {
  id("com.android.application")
  id("org.jetbrains.kotlin.android")
}

android {
  namespace = "ai.carrotpilot.external"
  compileSdk = 35

  defaultConfig {
    applicationId = "ai.carrotpilot.external"
    minSdk = 29
    targetSdk = 35
    versionCode = 2
    versionName = "0.2.0"
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
}

dependencies {
  implementation("com.microsoft.onnxruntime:onnxruntime-android:1.24.3")
}
