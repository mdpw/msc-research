package com.example.voiceassistant

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {

    private lateinit var voskService: VoskService
    private lateinit var audioRecorder: AudioRecorder
    private val TAG = "MainActivity"

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        Log.d(TAG, "Permission result: $isGranted")
        if (!isGranted) {
            // Handle permission denied
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Request permission
        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }

        // Initialize services
        voskService = VoskService(this)
        audioRecorder = AudioRecorder(this)

        setContent {
            VoiceAssistantScreen(
                voskService = voskService,
                audioRecorder = audioRecorder,
                lifecycleScope = lifecycleScope
            )
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        voskService.release()
    }
}

@Composable
fun VoiceAssistantScreen(
    voskService: VoskService,
    audioRecorder: AudioRecorder,
    lifecycleScope: kotlinx.coroutines.CoroutineScope
) {
    var status by remember { mutableStateOf("Initializing...") }
    var transcription by remember { mutableStateOf("") }
    var isRecording by remember { mutableStateOf(false) }
    var isInitialized by remember { mutableStateOf(false) }

    val context = LocalContext.current
    val TAG = "VoiceAssistantScreen"

    // Initialize on first composition
    LaunchedEffect(Unit) {
        try {
            voskService.initialize()
            status = "Ready to record!"
            isInitialized = true
        } catch (e: Exception) {
            Log.e(TAG, "Initialization error: ${e.message}", e)
            status = "Error: ${e.message}"
        }
    }

    Surface(
        modifier = Modifier.fillMaxSize(),
        color = MaterialTheme.colorScheme.background
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(24.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center
        ) {
            Text(
                text = "Hotel Voice Assistant",
                style = MaterialTheme.typography.headlineMedium,
                modifier = Modifier.padding(bottom = 32.dp)
            )

            Button(
                onClick = {
                    try {
                        if (isRecording) {
                            Log.d(TAG, "🛑 Stop button clicked")
                            // Stop recording
                            isRecording = false
                            status = "Transcribing..."

                            lifecycleScope.launch {
                                try {
                                    val audioData = audioRecorder.stopRecording()
                                    Log.d(TAG, "Got ${audioData.size} audio samples")

                                    if (audioData.isEmpty()) {
                                        transcription = "No audio recorded"
                                        status = "Ready to record!"
                                    } else {
                                        val text = voskService.transcribeAudio(audioData)

                                        if (text.isNotEmpty()) {
                                            transcription = "You said: $text"
                                            status = "Ready to record!"
                                        } else {
                                            transcription = "Could not transcribe (empty result)"
                                            status = "Try again"
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e(TAG, "Error in transcription: ${e.message}", e)
                                    transcription = "Error: ${e.message}"
                                    status = "Error occurred"
                                }
                            }
                        } else {
                            Log.d(TAG, "▶️ Start button clicked")
                            // Start recording
                            if (ContextCompat.checkSelfPermission(
                                    context,
                                    Manifest.permission.RECORD_AUDIO
                                ) == PackageManager.PERMISSION_GRANTED
                            ) {
                                audioRecorder.startRecording()
                                isRecording = true
                                status = "Recording... Speak now!"
                                transcription = ""
                            } else {
                                status = "Microphone permission not granted"
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Button click error: ${e.message}", e)
                        status = "Error: ${e.message}"
                        isRecording = false
                    }
                },
                enabled = isInitialized,
                modifier = Modifier.size(200.dp)
            ) {
                Text(
                    text = if (isRecording) "Stop Recording" else "Start Recording",
                    style = MaterialTheme.typography.titleMedium
                )
            }

            Spacer(modifier = Modifier.height(24.dp))

            Card(
                modifier = Modifier.fillMaxWidth()
            ) {
                Text(
                    text = status,
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyLarge
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            if (transcription.isNotEmpty()) {
                Card(
                    modifier = Modifier.fillMaxWidth()
                ) {
                    Text(
                        text = transcription,
                        modifier = Modifier.padding(16.dp),
                        style = MaterialTheme.typography.bodyLarge
                    )
                }
            }
        }
    }
}