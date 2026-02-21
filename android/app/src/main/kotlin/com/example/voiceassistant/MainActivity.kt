package com.example.voiceassistant

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.speech.tts.TextToSpeech
import android.speech.tts.UtteranceProgressListener
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
import androidx.compose.material.icons.Icons
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Refresh
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
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.launch
import java.util.*

data class RequestItem(
    val id: Int,
    val requestText: String,
    val intent: String,
    val confidence: Float,
    val department: String,
    val status: String,
    val timestamp: String
)

class MainActivity : ComponentActivity() {
    private lateinit var voskService: VoskService
    private lateinit var audioRecorder: AudioRecorder
    private lateinit var nluService: NLUService
    private val _apiService = mutableStateOf<ApiService?>(null)
    private var apiService: ApiService
        get() = _apiService.value!!
        set(value) { _apiService.value = value }
    private lateinit var tts: TextToSpeech
    private lateinit var webSocketService: WebSocketService
    private var ttsReady = false
    private val TAG = "MainActivity"
    private val ROOM_NUMBER = "101"
    private val _requestHistory = mutableStateListOf<RequestItem>()

    // Network & server config state
    private val _wifiSsid = mutableStateOf<String?>(null)
    private val _deviceIp = mutableStateOf<String?>(null)
    private val _activeProfileIndex = mutableIntStateOf(0)
    private val _profiles = mutableStateListOf<NetworkProfile>()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            initializeServices()
        } else {
            Toast.makeText(this, "Microphone permission is required.", Toast.LENGTH_LONG).show()
        }
    }

    private val locationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            refreshNetworkInfo()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Load server config from SharedPreferences
        ServerConfig.load(this)
        _profiles.clear()
        _profiles.addAll(ServerConfig.profiles)
        _activeProfileIndex.intValue = ServerConfig.activeProfileIndex

        voskService = VoskService(this)
        audioRecorder = AudioRecorder(this)
        nluService = NLUService(this)
        apiService = ApiService(ServerConfig.baseUrl)

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts.setLanguage(Locale.US)
                if (result == TextToSpeech.LANG_MISSING_DATA || result == TextToSpeech.LANG_NOT_SUPPORTED) {
                    Log.e(TAG, "TTS: Language not supported")
                } else {
                    tts.setSpeechRate(0.9f)
                    tts.setPitch(1.0f)
                    ttsReady = true
                    Log.d(TAG, "✅ TTS Initialized")
                }
            }
        }

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            initializeServices()
        } else {
            requestPermissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
        }

        // Request location permission for WiFi SSID detection
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) == PackageManager.PERMISSION_GRANTED) {
            refreshNetworkInfo()
        } else {
            locationPermissionLauncher.launch(Manifest.permission.ACCESS_FINE_LOCATION)
        }

        connectWebSocket()

        setContent {
            VoiceAssistantScreen(
                roomNumber = ROOM_NUMBER,
                voskService = voskService,
                audioRecorder = audioRecorder,
                nluService = nluService,
                apiService = _apiService.value ?: return@setContent,
                lifecycleScope = lifecycleScope,
                requestHistory = _requestHistory,
                wifiSsid = _wifiSsid,
                deviceIp = _deviceIp,
                profiles = _profiles,
                activeProfileIndex = _activeProfileIndex,
                onSpeakAndWait = { message -> speakAndWait(message) },
                onSpeakResponse = { message -> speakFire(message) },
                onAddRequest = { request -> _requestHistory.add(0, request) },
                onCloseApp = { finish() },
                onRefreshRequests = { refreshRequests() },
                onProfileSwitch = { index -> switchProfile(index) },
                onProfileUpdate = { index, profile -> updateProfile(index, profile) },
                onProfileAdd = { profile -> addProfile(profile) },
                onProfileRemove = { index -> removeProfile(index) }
            )
        }

        refreshRequests()
    }

    private fun refreshRequests() {
        apiService.getRequestHistory(ROOM_NUMBER,
            onSuccess = { history ->
                runOnUiThread {
                    _requestHistory.clear()
                    _requestHistory.addAll(history)
                }
            },
            onError = { error ->
                runOnUiThread {
                    Toast.makeText(this, "Sync failed", Toast.LENGTH_SHORT).show()
                    speakFire("There is an issue with network connectivity. Please contact the help desk using the land line phone.")
                }
            }
        )
    }

    private fun initializeServices() {
        Log.d(TAG, "🚀 Initializing services...")
        lifecycleScope.launch {
            voskService.initialize()
        }
        nluService.initialize()
    }

    private fun refreshNetworkInfo() {
        _wifiSsid.value = NetworkUtils.getWifiSsid(this)
        _deviceIp.value = NetworkUtils.getDeviceIp(this)
        Log.d(TAG, "Network: SSID=${_wifiSsid.value}, IP=${_deviceIp.value}")
    }

    private fun switchProfile(index: Int) {
        ServerConfig.switchProfile(this, index)
        _activeProfileIndex.intValue = index
        reconnectToServer()
        Toast.makeText(this, "Switched to: ${ServerConfig.activeProfile.name}", Toast.LENGTH_SHORT).show()
    }

    private fun updateProfile(index: Int, profile: NetworkProfile) {
        ServerConfig.updateProfile(this, index, profile)
        _profiles.clear()
        _profiles.addAll(ServerConfig.profiles)
        if (index == _activeProfileIndex.intValue) {
            reconnectToServer()
        }
    }

    private fun addProfile(profile: NetworkProfile) {
        ServerConfig.addProfile(this, profile)
        _profiles.clear()
        _profiles.addAll(ServerConfig.profiles)
    }

    private fun removeProfile(index: Int) {
        ServerConfig.removeProfile(this, index)
        _profiles.clear()
        _profiles.addAll(ServerConfig.profiles)
        _activeProfileIndex.intValue = ServerConfig.activeProfileIndex
        reconnectToServer()
    }

    private fun speakFire(message: String) {
        if (ttsReady) {
            val params = Bundle()
            params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "fire")
            tts.speak(message, TextToSpeech.QUEUE_FLUSH, params, "fire")
        }
    }

    private suspend fun speakAndWait(message: String) {
        if (!ttsReady) return
        val deferred = CompletableDeferred<Unit>()
        tts.setOnUtteranceProgressListener(object : UtteranceProgressListener() {
            override fun onStart(utteranceId: String?) {}
            override fun onDone(utteranceId: String?) { deferred.complete(Unit) }
            @Deprecated("Deprecated in Java")
            override fun onError(utteranceId: String?) { deferred.complete(Unit) }
        })
        val params = Bundle()
        params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "greeting")
        tts.speak(message, TextToSpeech.QUEUE_FLUSH, params, "greeting")
        deferred.await()
    }

    private fun reconnectToServer() {
        apiService = ApiService(ServerConfig.baseUrl)
        if (::webSocketService.isInitialized) {
            webSocketService.disconnect()
        }
        connectWebSocket()
        // Check connectivity by fetching history
        apiService.getRequestHistory(ROOM_NUMBER,
            onSuccess = { history ->
                runOnUiThread {
                    _requestHistory.clear()
                    _requestHistory.addAll(history)
                    speakFire("Successfully connected to ${ServerConfig.activeProfile.name}. System is ready.")
                }
            },
            onError = { error ->
                runOnUiThread {
                    speakFire("Failed to connect to ${ServerConfig.activeProfile.name}. Please check the network connection or contact the help desk using the land line phone.")
                }
            }
        )
        Log.d(TAG, "Server config updated: ${ServerConfig.baseUrl}")
    }

    private fun connectWebSocket() {
        try {
            webSocketService = WebSocketService(ROOM_NUMBER, ServerConfig.wsUrl(ROOM_NUMBER))
            webSocketService.connect(
                onMessage = { message ->
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, message, Toast.LENGTH_LONG).show()
                    }
                },
                onStatusChange = { requestId, status ->
                    Log.d(TAG, "🔌 WebSocket status change: request $requestId -> $status")
                    runOnUiThread {
                        val index = _requestHistory.indexOfFirst { it.id == requestId }
                        if (index != -1) {
                            val oldRequest = _requestHistory[index]
                            _requestHistory[index] = oldRequest.copy(status = status)

                            if (ttsReady) {
                                val statusMessage = when (status) {
                                    "in_progress" -> "Your request No.$requestId is now being processed."
                                    "completed" -> "Your request No.$requestId is completed. Thank you for your patience!"
                                    else -> "Your request No.$requestId is now $status."
                                }
                                val params = Bundle()
                                params.putString(TextToSpeech.Engine.KEY_PARAM_UTTERANCE_ID, "StatusUpdate_$requestId")
                                tts.speak(statusMessage, TextToSpeech.QUEUE_FLUSH, params, "StatusUpdate_$requestId")
                            }
                        } else {
                            refreshRequests()
                        }
                    }
                }
            )
        } catch (e: Exception) {
            Log.e(TAG, "WebSocket error", e)
            runOnUiThread {
                speakFire("There is an issue with network connectivity. Please contact the help desk using the land line phone.")
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        tts.shutdown()
        voskService.release()
        nluService.close()
        webSocketService.disconnect()
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
    wifiSsid: State<String?>,
    deviceIp: State<String?>,
    profiles: List<NetworkProfile>,
    activeProfileIndex: MutableIntState,
    onSpeakAndWait: suspend (String) -> Unit,
    onSpeakResponse: (String) -> Unit,
    onAddRequest: (RequestItem) -> Unit,
    onCloseApp: () -> Unit,
    onRefreshRequests: () -> Unit,
    onProfileSwitch: (Int) -> Unit,
    onProfileUpdate: (Int, NetworkProfile) -> Unit,
    onProfileAdd: (NetworkProfile) -> Unit,
    onProfileRemove: (Int) -> Unit
) {
    val context = LocalContext.current
    var isRecording by remember { mutableStateOf(false) }
    var isProcessing by remember { mutableStateOf(false) }
    var statusMessage by remember { mutableStateOf("Tap microphone to start") }
    var lastTranscription by remember { mutableStateOf("") }
    var lastIntent by remember { mutableStateOf("") }
    var lastConfidence by remember { mutableStateOf(0f) }
    var lastDepartment by remember { mutableStateOf("") }
    var showServerDialog by remember { mutableStateOf(false) }
    var settingsExpanded by remember { mutableStateOf(false) }

    MaterialTheme {
        Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
            Column(modifier = Modifier.fillMaxSize().padding(12.dp)) {

                // ── Collapsible Settings Panel (About + Network) ──
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
                ) {
                    Column(modifier = Modifier.fillMaxWidth()) {
                        // Always-visible header row
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column {
                                Text(text = "Sera - Voice Assistant", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold)
                                Text(text = "Room $roomNumber", style = MaterialTheme.typography.bodySmall)
                            }
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                IconButton(onClick = { settingsExpanded = !settingsExpanded }, modifier = Modifier.size(32.dp)) {
                                    Icon(
                                        if (settingsExpanded) Icons.Default.KeyboardArrowUp else Icons.Default.KeyboardArrowDown,
                                        contentDescription = "Toggle Settings"
                                    )
                                }
                                IconButton(onClick = onCloseApp, modifier = Modifier.size(32.dp)) {
                                    Icon(Icons.Default.Close, contentDescription = "Close App", tint = MaterialTheme.colorScheme.error)
                                }
                            }
                        }

                        // Collapsible content: Network info + Profile selector
                        AnimatedVisibility(visible = settingsExpanded) {
                            Card(
                                modifier = Modifier.fillMaxWidth().padding(horizontal = 8.dp, vertical = 4.dp),
                                colors = CardDefaults.cardColors(containerColor = Color(0xFFE8F5E9))
                            ) {
                                Column(modifier = Modifier.fillMaxWidth().padding(10.dp)) {
                                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                        Text(
                                            text = "WiFi: ${wifiSsid.value ?: "Not connected"}",
                                            style = MaterialTheme.typography.bodySmall,
                                            fontWeight = FontWeight.Medium
                                        )
                                        Text(
                                            text = "IP: ${deviceIp.value ?: "N/A"}",
                                            style = MaterialTheme.typography.bodySmall
                                        )
                                    }
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Row(
                                        modifier = Modifier.fillMaxWidth(),
                                        horizontalArrangement = Arrangement.spacedBy(6.dp),
                                        verticalAlignment = Alignment.CenterVertically
                                    ) {
                                        profiles.forEachIndexed { index, profile ->
                                            val isActive = index == activeProfileIndex.intValue
                                            FilterChip(
                                                selected = isActive,
                                                onClick = { if (!isActive) onProfileSwitch(index) },
                                                label = { Text(profile.name, fontSize = 11.sp) },
                                                modifier = Modifier.weight(1f)
                                            )
                                        }
                                        IconButton(onClick = { showServerDialog = true }, modifier = Modifier.size(28.dp)) {
                                            Icon(Icons.Default.Edit, contentDescription = "Edit Profiles", modifier = Modifier.size(16.dp))
                                        }
                                    }
                                    val active = profiles.getOrNull(activeProfileIndex.intValue)
                                    if (active != null) {
                                        Text(
                                            text = "Server: ${active.serverIp}:${active.serverPort}",
                                            style = MaterialTheme.typography.bodySmall,
                                            color = Color(0xFF2E7D32)
                                        )
                                    }
                                }
                            }
                        }
                    }
                }

                // Profile edit dialog
                if (showServerDialog) {
                    ProfileEditDialog(
                        profiles = profiles,
                        activeIndex = activeProfileIndex.intValue,
                        onDismiss = { showServerDialog = false },
                        onUpdateProfile = { index, profile -> onProfileUpdate(index, profile) },
                        onAddProfile = { profile -> onProfileAdd(profile) },
                        onRemoveProfile = { index ->
                            onProfileRemove(index)
                            if (profiles.size <= 1) showServerDialog = false
                        }
                    )
                }

                Spacer(modifier = Modifier.height(6.dp))

                // ── Mic Section (1/3 of remaining space) ──
                Card(
                    modifier = Modifier.fillMaxWidth().weight(1f),
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)
                ) {
                    Column(
                        modifier = Modifier.fillMaxSize().padding(12.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Center
                    ) {
                        Button(
                            onClick = {
                                if (!isRecording && !isProcessing) {
                                    lifecycleScope.launch {
                                        statusMessage = "Hello! I'm Sera. How can I help you?"
                                        onSpeakAndWait("Hello! I'm Sera. How can I help you?")
                                        processVoiceRequest(
                                            audioRecorder, voskService, nluService, apiService, roomNumber, lifecycleScope,
                                            { isRecording = true }, { isRecording = false }, { isProcessing = true }, { isProcessing = false },
                                            { statusMessage = it }, { lastTranscription = it },
                                            { i, c -> lastIntent = i; lastConfidence = c },
                                            onSpeakResponse,
                                            { request -> lastDepartment = request.department; onAddRequest(request) },
                                            { statusMessage = "Tap microphone to start" }
                                        )
                                    }
                                }
                            },
                            enabled = !isRecording && !isProcessing,
                            modifier = Modifier.size(80.dp), shape = RoundedCornerShape(50),
                            colors = ButtonDefaults.buttonColors(
                                containerColor = when {
                                    isRecording -> MaterialTheme.colorScheme.error
                                    isProcessing -> MaterialTheme.colorScheme.tertiary
                                    else -> MaterialTheme.colorScheme.primary
                                }
                            )
                        ) {
                            val buttonIcon = if (isRecording) "⏺️" else if (isProcessing) "⚙️" else "🎤"
                            Text(text = buttonIcon, fontSize = 32.sp)
                        }

                        Spacer(modifier = Modifier.height(4.dp))
                        Text(
                            text = when {
                                isRecording -> "Recording..."
                                isProcessing -> "Processing..."
                                else -> "Tap to speak"
                            },
                            style = MaterialTheme.typography.bodySmall
                        )

                        if (lastTranscription.isNotEmpty()) {
                            Spacer(modifier = Modifier.height(6.dp))
                            HorizontalDivider()
                            Spacer(modifier = Modifier.height(6.dp))
                            Text(text = "Transcription:", style = MaterialTheme.typography.bodySmall)
                            Text(text = "\"$lastTranscription\"", fontWeight = FontWeight.Medium, fontSize = 13.sp)
                            if (lastIntent.isNotEmpty()) {
                                Spacer(modifier = Modifier.height(2.dp))
                                Text(text = "Intent: $lastIntent (${(lastConfidence * 100).toInt()}%)", style = MaterialTheme.typography.bodySmall)
                            }
                            if (lastDepartment.isNotEmpty()) {
                                Text(text = "Department: $lastDepartment", style = MaterialTheme.typography.bodySmall)
                            }
                        }
                    }
                }

                Spacer(modifier = Modifier.height(6.dp))

                // ── Recent Requests Section (2/3 of remaining space) ──
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(text = "Recent Requests", style = MaterialTheme.typography.titleSmall, fontWeight = FontWeight.Bold)
                    IconButton(onClick = onRefreshRequests, modifier = Modifier.size(28.dp)) {
                        Icon(Icons.Default.Refresh, contentDescription = "Refresh", modifier = Modifier.size(18.dp))
                    }
                }

                Spacer(modifier = Modifier.height(4.dp))

                Box(modifier = Modifier.fillMaxWidth().weight(2f)) {
                    if (requestHistory.isEmpty()) {
                        Card(modifier = Modifier.fillMaxSize(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceVariant)) {
                            Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                    Text(text = "📝", fontSize = 40.sp)
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(text = "No requests yet", style = MaterialTheme.typography.bodySmall)
                                }
                            }
                        }
                    } else {
                        LazyColumn(modifier = Modifier.fillMaxSize(), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            items(requestHistory) { request -> RequestCard(request = request) }
                        }
                    }
                }
            }
        }
    }
}

suspend fun processVoiceRequest(
    audioRecorder: AudioRecorder,
    voskService: VoskService,
    nluService: NLUService,
    apiService: ApiService,
    roomNumber: String,
    lifecycleScope: kotlinx.coroutines.CoroutineScope,
    onRecordingStart: () -> Unit,
    onRecordingStop: () -> Unit,
    onProcessingStart: () -> Unit,
    onProcessingStop: () -> Unit,
    onStatusUpdate: (String) -> Unit,
    onTranscriptionUpdate: (String) -> Unit,
    onIntentUpdate: (String, Float) -> Unit,
    onSpeakResponse: (String) -> Unit,
    onAddRequest: (RequestItem) -> Unit,
    onComplete: () -> Unit
) {
    if (ContextCompat.checkSelfPermission(audioRecorder.getContext(), Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
        onStatusUpdate("Permission denied")
        onComplete()
        return
    }

    try {
        onRecordingStart(); onStatusUpdate("Listening... (speak now)")

        val audioData = audioRecorder.recordWithVAD(
            silenceTimeoutMs = 1500L,
            maxDurationMs = 10000L,
            onStateChange = { state -> onStatusUpdate(state) }
        )

        onRecordingStop(); onProcessingStart(); onStatusUpdate("Processing...")

        lifecycleScope.launch {
            try {
                onStatusUpdate("Transcribing...")
                val transcription = voskService.transcribeAudio(audioData)
                onTranscriptionUpdate(transcription)

                if (transcription.isEmpty()) {
                    onStatusUpdate("No speech detected"); onProcessingStop(); onComplete(); return@launch
                }

                onStatusUpdate("Understanding...")
                val cleanedTranscription = cleanTranscription(transcription)
                val intentResult = nluService.classifyIntent(cleanedTranscription)
                onIntentUpdate(intentResult.name, intentResult.confidence)

                onStatusUpdate("Submitting...")
                apiService.submitRequest(
                    roomNumber = roomNumber, requestText = transcription, intent = intentResult.name,
                    onSuccess = { response ->
                        onAddRequest(RequestItem(response.requestId, transcription, intentResult.name, intentResult.confidence, "Routing...", "pending", getCurrentTime()))
                        val speechText = "Your request No.${response.requestId} has been received."
                        onSpeakResponse(speechText)
                        onProcessingStop(); onComplete()
                    },
                    onError = {
                        onStatusUpdate("Submission failed")
                        onSpeakResponse("Sorry, I could not send your request due to a network issue. Please contact the help desk using the land line phone.")
                        onProcessingStop(); onComplete()
                    }
                )
            } catch (e: Exception) { onStatusUpdate("Error occurred"); onProcessingStop(); onComplete() }
        }
    } catch (e: Exception) { onRecordingStop(); onProcessingStop(); onComplete() }
}

@Composable
fun RequestCard(request: RequestItem) {
    Card(modifier = Modifier.fillMaxWidth(), elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)) {
        Column(modifier = Modifier.fillMaxWidth().padding(12.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Text(text = "Request No.${request.id}", fontWeight = FontWeight.Bold)
                StatusBadge(status = request.status)
            }
            Spacer(modifier = Modifier.height(8.dp)); Text(text = request.requestText); Spacer(modifier = Modifier.height(4.dp))
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
                Text(text = "${request.intent} | ${request.department}", style = MaterialTheme.typography.bodySmall)
                Text(text = request.timestamp, style = MaterialTheme.typography.bodySmall)
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
    Box(modifier = Modifier.background(backgroundColor, RoundedCornerShape(12.dp)).padding(horizontal = 12.dp, vertical = 4.dp)) {
        Text(text = text, color = textColor, fontSize = 12.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
fun ProfileEditDialog(
    profiles: List<NetworkProfile>,
    activeIndex: Int,
    onDismiss: () -> Unit,
    onUpdateProfile: (Int, NetworkProfile) -> Unit,
    onAddProfile: (NetworkProfile) -> Unit,
    onRemoveProfile: (Int) -> Unit
) {
    var editingIndex by remember { mutableStateOf<Int?>(null) }
    var editName by remember { mutableStateOf("") }
    var editIp by remember { mutableStateOf("") }
    var editPort by remember { mutableStateOf("") }

    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Network Profiles") },
        text = {
            Column {
                profiles.forEachIndexed { index, profile ->
                    if (editingIndex == index) {
                        OutlinedTextField(
                            value = editName,
                            onValueChange = { editName = it },
                            label = { Text("Name") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        OutlinedTextField(
                            value = editIp,
                            onValueChange = { editIp = it },
                            label = { Text("Server IP") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )
                        Spacer(modifier = Modifier.height(4.dp))
                        OutlinedTextField(
                            value = editPort,
                            onValueChange = { editPort = it },
                            label = { Text("Port") },
                            singleLine = true,
                            modifier = Modifier.fillMaxWidth()
                        )
                        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                            TextButton(onClick = {
                                onUpdateProfile(index, NetworkProfile(editName.trim(), editIp.trim(), editPort.toIntOrNull() ?: 8000))
                                editingIndex = null
                            }) { Text("Save") }
                            TextButton(onClick = { editingIndex = null }) { Text("Cancel") }
                        }
                    } else {
                        Row(
                            modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                            horizontalArrangement = Arrangement.SpaceBetween,
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            Column(modifier = Modifier.weight(1f)) {
                                Text(
                                    text = profile.name + if (index == activeIndex) " (Active)" else "",
                                    fontWeight = if (index == activeIndex) FontWeight.Bold else FontWeight.Normal,
                                    fontSize = 14.sp
                                )
                                Text(
                                    text = "${profile.serverIp}:${profile.serverPort}",
                                    fontSize = 12.sp,
                                    color = Color.Gray
                                )
                            }
                            IconButton(onClick = {
                                editingIndex = index
                                editName = profile.name
                                editIp = profile.serverIp
                                editPort = profile.serverPort.toString()
                            }, modifier = Modifier.size(32.dp)) {
                                Icon(Icons.Default.Edit, contentDescription = "Edit", modifier = Modifier.size(16.dp))
                            }
                            if (profiles.size > 1) {
                                IconButton(onClick = { onRemoveProfile(index) }, modifier = Modifier.size(32.dp)) {
                                    Icon(Icons.Default.Close, contentDescription = "Remove", modifier = Modifier.size(16.dp), tint = Color.Red)
                                }
                            }
                        }
                    }
                    if (index < profiles.lastIndex) HorizontalDivider()
                }
            }
        },
        confirmButton = {
            TextButton(onClick = {
                onAddProfile(NetworkProfile("New Network", "192.168.1.100", 8000))
            }) { Text("Add Profile") }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) { Text("Close") }
        }
    )
}

fun cleanTranscription(text: String): String {
    val prefixPatterns = listOf(
        "hi sera", "hey sera", "hello sera", "sera",
        "hi there", "hey there", "hello there",
        "hi", "hey", "hello",
        "excuse me", "please", "can you", "could you", "i need you to"
    )
    var cleaned = text.lowercase().trim()
    for (pattern in prefixPatterns) {
        if (cleaned.startsWith(pattern)) {
            cleaned = cleaned.removePrefix(pattern).trim()
            cleaned = cleaned.removePrefix(",").removePrefix(".").trim()
            break
        }
    }
    return cleaned.ifEmpty { text.trim() }
}

fun getCurrentTime(): String {
    val calendar = Calendar.getInstance()
    return String.format(Locale.getDefault(), "%02d:%02d", calendar.get(Calendar.HOUR_OF_DAY), calendar.get(Calendar.MINUTE))
}

fun AudioRecorder.getContext(): android.content.Context {
    val field = this.javaClass.getDeclaredField("context")
    field.isAccessible = true
    return field.get(this) as android.content.Context
}
