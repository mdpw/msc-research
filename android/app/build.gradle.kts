plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.voiceassistant"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.voiceassistant"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        // ✅ Add NDK filters for Whisper.cpp
        ndk {
            abiFilters += listOf("armeabi-v7a", "arm64-v8a", "x86", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = "11"
    }

    buildFeatures {
        compose = true
    }

    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
    }
}

dependencies {
    // Compose dependencies
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.compose.material3)

    // Test dependencies
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    debugImplementation(libs.androidx.compose.ui.tooling)
    debugImplementation(libs.androidx.compose.ui.test.manifest)

    // AppCompat
    implementation("androidx.appcompat:appcompat:1.6.1")

    // Coroutines
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")

    // ============================================
    // SPEECH-TO-TEXT (STT) OPTIONS
    // ============================================

    // Option 1: VOSK (Current - Lower accuracy ~75-85%)
    // ❌ COMMENT OUT OR REMOVE when migrating to Whisper
    implementation("com.alphacephei:vosk-android:0.3.32")
    implementation("net.java.dev.jna:jna:5.13.0@aar")

    // Option 2: WHISPER.CPP (Recommended - Higher accuracy ~90-95%)
    // ✅ UNCOMMENT when ready to migrate
    //implementation("com.whispercpp:whisper:1.5.5")
    //implementation("net.java.dev.jna:jna:5.13.0@aar")

    // ============================================
    // NLU (Natural Language Understanding)
    // ============================================

    // TensorFlow Lite for MobileBERT
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    implementation("org.tensorflow:tensorflow-lite-support:0.5.0")

    // Optional: TFLite GPU delegate (for better performance)
    // implementation("org.tensorflow:tensorflow-lite-gpu:2.17.0")

    // ============================================
    // WEBSOCKET & NETWORKING
    // ============================================

    // OkHttp for WebSocket connections (status updates from backend)
    implementation("com.squareup.okhttp3:okhttp:4.11.0")
}