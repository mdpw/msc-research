package com.example.voiceassistant

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.launch
import java.util.*

// Data class for request history
data class RequestItem(
    val id: Int,
    val requestText: String,
    val intent: String,
    val status: String,
    val timestamp: String
)

class MainActivity : ComponentActivity() {

    private lateinit var voskService: VoskService
    private lateinit var audioRecorder: AudioRecorder
    private lateinit var nluService: NLUService
    private lateinit var apiService: ApiService
    private lateinit var tts: TextToSpeech
    private lateinit var webSocketService: WebSocketService
    private var ttsReady = false
    private val TAG = "MainActivity"

    // Fixed room number for device
    private val ROOM_NUMBER = "101"

    // Mutable state for request history that can be updated from WebSocket
    private val _requestHistory = mutableStateListOf<RequestItem>()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        Log.d(TAG, "Permission result: $isGranted")
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
        nluService = NLUService(this)
        apiService = ApiService()

        // Initialize TTS
        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts.setLanguage(Locale.US)
                ttsReady = result != TextToSpeech.LANG_MISSING_DATA &&
                        result != TextToSpeech.LANG_NOT_SUPPORTED
                if (ttsReady) {
                    Toast.makeText(this, "TTS Ready", Toast.LENGTH_SHORT).show()
                    Log.d(TAG, "✅ TTS initialized")
                }
            }
        }

        // Initialize WebSocket
        connectWebSocket()

        setContent {
            VoiceAssistantScreen(
                roomNumber = ROOM_NUMBER,
                voskService = voskService,
                audioRecorder = audioRecorder,
                nluService = nluService,
                apiService = apiService,
                lifecycleScope = lifecycleScope,
                requestHistory = _requestHistory,
                onSpeakResponse = { message -> speakResponse(message) },
                onAddRequest = { request -> _requestHistory.add(0, request) }
            )
        }
    }

    private fun connectWebSocket() {
        webSocketService = WebSocketService(ROOM_NUMBER)
        webSocketService.connect(
            onMessage = { message ->
                runOnUiThread {
                    speakResponse(message)
                    Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                }
            },
            onStatusChange = { requestId, status ->
                runOnUiThread {
                    // Update request status in history
                    val index = _requestHistory.indexOfFirst { it.id == requestId }
                    if (index != -1) {
                        val oldRequest = _requestHistory[index]
                        _requestHistory[index] = oldRequest.copy(status = status)
                        Log.d(TAG, "📱 Request $requestId status updated to: $status")
                    }
                }
            }
        )
    }

    private fun speakResponse(message: String) {
        if (ttsReady) {
            tts.speak(message, TextToSpeech.QUEUE_FLUSH, null, null)
            Log.d(TAG, "🔊 Speaking: $message")
        } else {
            Toast.makeText(this, "TTS not ready", Toast.LENGTH_SHORT).show()
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        voskService.release()
        nluService.release()

        if (::tts.isInitialized) {
            tts.stop()
            tts.shutdown()
        }

        if (::webSocketService.isInitialized) {
            webSocketService.disconnect()
        }
    }
}

@Composable
fun VoiceAssistantScreen(
    roomNumber: String,
    voskService: VoskService,
    audioRecorder: AudioRecorder,
    nluService: NLUService,
    apiService: ApiService,
    lifecycleScope: kotlinx.coroutines.CoroutineScope,
    requestHistory: List<RequestItem>,
    onSpeakResponse: (String) -> Unit,
    onAddRequest: (RequestItem) -> Unit
) {
    var status by remember { mutableStateOf("Initializing...") }
    var isRecording by remember { mutableStateOf(false) }
    var isInitialized by remember { mutableStateOf(false) }

    val context = LocalContext.current
    val TAG = "VoiceAssistantScreen"

    // Initialize STT and NLU
    LaunchedEffect(Unit) {
        try {
            status = "Initializing speech recognition..."
            voskService.initialize()

            status = "Initializing NLU..."
            nluService.initialize()

            status = "Ready! Press the button to speak"
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
                .padding(16.dp)
        ) {
            // Header
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            ) {
                Column(
                    modifier = Modifier.padding(16.dp),
                    horizontalAlignment = Alignment.CenterHorizontally
                ) {
                    Text(
                        text = "🏨 Hotel Voice Assistant",
                        style = MaterialTheme.typography.headlineSmall,
                        fontWeight = FontWeight.Bold
                    )
                    Spacer(modifier = Modifier.height(8.dp))
                    Text(
                        text = "Room $roomNumber",
                        style = MaterialTheme.typography.titleMedium,
                        color = MaterialTheme.colorScheme.primary
                    )
                }
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Recording Button
            Button(
                onClick = {
                    try {
                        if (isRecording) {
                            Log.d(TAG, "🛑 Stop recording")
                            isRecording = false
                            status = "Processing..."

                            lifecycleScope.launch {
                                try {
                                    val audioData = audioRecorder.stopRecording()
                                    Log.d(TAG, "Got ${audioData.size} audio samples")

                                    if (audioData.isEmpty()) {
                                        status = "No audio recorded. Try again."
                                    } else {
                                        status = "Transcribing..."
                                        val text = voskService.transcribeAudio(audioData)

                                        if (text.isNotEmpty()) {
                                            status = "Understanding request..."
                                            val intent = nluService.classifyIntent(text)

                                            status = "Sending to backend..."
                                            apiService.submitRequest(
                                                roomNumber = roomNumber,
                                                requestText = text,
                                                intent = intent.name,
                                                onSuccess = { response ->
                                                    lifecycleScope.launch {
                                                        // Add to history
                                                        val newRequest = RequestItem(
                                                            id = response.requestId,
                                                            requestText = text,
                                                            intent = intent.name,
                                                            status = "pending",
                                                            timestamp = getCurrentTime()
                                                        )
                                                        onAddRequest(newRequest)

                                                        status = "✅ Request sent!"
                                                        onSpeakResponse(response.message)
                                                    }
                                                },
                                                onError = { error ->
                                                    lifecycleScope.launch {
                                                        status = "❌ Error: $error"
                                                    }
                                                }
                                            )

                                        } else {
                                            status = "Could not understand. Try again."
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e(TAG, "Error: ${e.message}", e)
                                    status = "Error: ${e.message}"
                                }
                            }
                        } else {
                            Log.d(TAG, "▶️ Start recording")
                            if (ContextCompat.checkSelfPermission(
                                    context,
                                    Manifest.permission.RECORD_AUDIO
                                ) == PackageManager.PERMISSION_GRANTED
                            ) {
                                audioRecorder.startRecording()
                                isRecording = true
                                status = "🎤 Listening... Speak now!"
                            } else {
                                status = "Microphone permission required"
                            }
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Error: ${e.message}", e)
                        status = "Error: ${e.message}"
                        isRecording = false
                    }
                },
                enabled = isInitialized,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(80.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (isRecording)
                        MaterialTheme.colorScheme.error
                    else
                        MaterialTheme.colorScheme.primary
                )
            ) {
                Text(
                    text = if (isRecording) "🛑 Stop Recording" else "🎤 Start Recording",
                    style = MaterialTheme.typography.titleLarge,
                    fontWeight = FontWeight.Bold
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Status Card
            Card(
                modifier = Modifier.fillMaxWidth(),
                colors = CardDefaults.cardColors(
                    containerColor = MaterialTheme.colorScheme.secondaryContainer
                )
            ) {
                Text(
                    text = status,
                    modifier = Modifier.padding(16.dp),
                    style = MaterialTheme.typography.bodyLarge
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            // Request History
            Text(
                text = "Recent Requests",
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold
            )

            Spacer(modifier = Modifier.height(8.dp))

            if (requestHistory.isEmpty()) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.surfaceVariant
                    )
                ) {
                    Box(
                        modifier = Modifier.fillMaxSize(),
                        contentAlignment = Alignment.Center
                    ) {
                        Text(
                            text = "No requests yet",
                            color = MaterialTheme.colorScheme.onSurfaceVariant
                        )
                    }
                }
            } else {
                LazyColumn(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    verticalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    items(requestHistory) { request ->
                        RequestCard(request = request)
                    }
                }
            }
        }
    }
}

@Composable
fun RequestCard(request: RequestItem) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Request #${request.id}",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold
                )
                StatusBadge(status = request.status)
            }

            Spacer(modifier = Modifier.height(8.dp))

            Text(
                text = request.requestText,
                style = MaterialTheme.typography.bodyMedium
            )

            Spacer(modifier = Modifier.height(4.dp))

            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text(
                    text = "Intent: ${request.intent}",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                Text(
                    text = request.timestamp,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}

@Composable
fun StatusBadge(status: String) {
    val (backgroundColor, textColor, text) = when (status) {
        "pending" -> Triple(Color(0xFFFFF3CD), Color(0xFF856404), "Pending")
        "in_progress" -> Triple(Color(0xFFD1ECF1), Color(0xFF0C5460), "In Progress")
        "completed" -> Triple(Color(0xFFD4EDDA), Color(0xFF155724), "Completed")
        else -> Triple(Color.Gray, Color.White, status)
    }

    Box(
        modifier = Modifier
            .background(backgroundColor, RoundedCornerShape(12.dp))
            .padding(horizontal = 12.dp, vertical = 4.dp)
    ) {
        Text(
            text = text,
            color = textColor,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

fun getCurrentTime(): String {
    val calendar = Calendar.getInstance()
    val hour = calendar.get(Calendar.HOUR_OF_DAY)
    val minute = calendar.get(Calendar.MINUTE)
    return String.format("%02d:%02d", hour, minute)
}