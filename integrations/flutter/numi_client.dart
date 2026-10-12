/// Minimal Flutter client for Numination → Supabase Edge Function `numi-chat`.
///
/// Do NOT call the Numi API URL from the app. Only call Supabase Functions
/// with the user JWT. The Edge Function talks to Numi with the server key.
///
/// Dependencies (pubspec.yaml):
///   supabase_flutter: ^2.0.0
///   http: ^1.2.0

library numi_client;

import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:supabase_flutter/supabase_flutter.dart';

class NumiChatResponse {
  final String conversationId;
  final String content;
  final Map<String, dynamic>? usage;

  NumiChatResponse({
    required this.conversationId,
    required this.content,
    this.usage,
  });

  factory NumiChatResponse.fromJson(Map<String, dynamic> json) {
    final message = json['message'] as Map<String, dynamic>? ?? {};
    return NumiChatResponse(
      conversationId: json['conversation_id'] as String? ?? '',
      content: message['content'] as String? ?? '',
      usage: json['usage'] as Map<String, dynamic>?,
    );
  }
}

class NumiClient {
  NumiClient({
    required this.supabaseUrl,
    this.functionName = 'numi-chat',
  });

  final String supabaseUrl;
  final String functionName;

  /// Send a user message. Pass [conversationId] to continue a thread.
  Future<NumiChatResponse> chat({
    required String message,
    String? conversationId,
    int maxTokens = 256,
    double temperature = 0.7,
  }) async {
    final session = Supabase.instance.client.auth.currentSession;
    if (session == null) {
      throw StateError('User must be signed in before calling Numi');
    }

    final uri = Uri.parse('$supabaseUrl/functions/v1/$functionName');
    final response = await http.post(
      uri,
      headers: {
        'Authorization': 'Bearer ${session.accessToken}',
        'apikey': Supabase.instance.client.rest.headers['apikey'] ?? '',
        'Content-Type': 'application/json',
      },
      body: jsonEncode({
        'message': message,
        if (conversationId != null) 'conversation_id': conversationId,
        'max_tokens': maxTokens,
        'temperature': temperature,
      }),
    );

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw Exception(
        'numi-chat failed: ${response.statusCode} ${response.body}',
      );
    }

    final body = jsonDecode(response.body) as Map<String, dynamic>;
    if (body['error'] != null) {
      throw Exception('numi-chat error: ${body['error']}');
    }
    return NumiChatResponse.fromJson(body);
  }
}

/// Example usage inside a widget:
///
/// ```dart
/// final client = NumiClient(supabaseUrl: 'https://YOUR.supabase.co');
/// final res = await client.chat(message: 'Hola');
/// print(res.content);
/// ```
