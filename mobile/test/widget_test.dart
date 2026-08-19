import 'package:flutter_test/flutter_test.dart';

import 'package:chatbot_asistan/main.dart';

void main() {
  testWidgets('Arayüz başlığı gösteriliyor', (WidgetTester tester) async {
    await tester.pumpWidget(const ChatAsistanApp());

    // AppBar başlığı ve örnek soru chip'leri ekranda olmalı.
    expect(find.text('CV Asistanı'), findsOneWidget);
    expect(find.text("BilgiTR'de hit-rate nedir?"), findsOneWidget);
  });
}
