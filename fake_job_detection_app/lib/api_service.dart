import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  // Use your computer's IP address if testing on a real device.
  static const String baseUrl = "http://127.0.0.1:5000";

  static Future<Map<String, dynamic>> checkJob(String jobDescription) async {
    final response = await http.post(
      Uri.parse("$baseUrl/predict"),
      headers: {"Content-Type": "application/json"},
      body: jsonEncode({"job_description": jobDescription}),
    );

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Failed to get response from server");
    }
  }
}
