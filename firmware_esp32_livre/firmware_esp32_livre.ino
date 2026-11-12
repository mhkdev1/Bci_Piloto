/*
 * firmware_esp32_livre.ino — segundo firmware da órtese: curso LIVRE (0..180°) e
 * controle completo pelo console (Monitor Serial da IDE ou qualquer terminal).
 *
 * Por que existe: a engrenagem do servo é menor que a engrenagem dos dedos (redução
 * para ganhar torque). Logo 90° no servo viram bem menos nos dedos, e o curso
 * 0..90° do firmware original não basta. Aqui o servo anda de 0 até o máximo que
 * ele alcança (180° nominais no S3003), e os limites podem ser ajustados sem regravar.
 *
 * LIGAÇÃO: igual ao firmware original (sinal GPIO 18, VCC em fonte 5 V externa,
 * GND comum).  Dependência: biblioteca "ESP32Servo".
 *
 * ---------------------------------------------------------------------------
 * PROTOCOLO — compatível com os scripts Python (1 byte, 1 eco "angle=<n>")
 * ---------------------------------------------------------------------------
 *   'L'        angle -= passo          'R'  angle += passo
 *   'C'        angle = REPOUSO         '0'..'9'  passo = dígito ('0' vira 1)
 *
 * ---------------------------------------------------------------------------
 * COMANDOS DE CONSOLE — letra + número + Enter (CR, LF ou CRLF são ignorados)
 * ---------------------------------------------------------------------------
 *   A<n>       vai direto ao ângulo n            ex.: A135
 *   P<n>       passo em graus, sem limite de 9   ex.: P15
 *   H<n>       repouso (para onde 'C' leva)      ex.: H0
 *   N<n>       ângulo mínimo permitido           ex.: N0
 *   X<n>       ângulo máximo permitido           ex.: X180
 *   U<a>,<b>   pulsos min,max em µs (calibração) ex.: U500,2500
 *   ?          imprime o estado
 *
 * No Monitor Serial deixe "Nova linha" ou "Ambos, NL e CR". Comandos de um byte
 * (L/R/C/dígitos) também funcionam no console, com ou sem Enter.
 *
 * CONFIG PELO PC: com config.SERVO_FIRMWARE_LIVRE = True, o nucleo/servo.py manda
 * N, X e H a cada conexão, a partir do config.py. Não precisa regravar para mudar
 * ângulos. H, N, X e U ficam salvos na flash (Preferences) e valem no próximo boot,
 * então o servo liga já no repouso certo, sem passar por 90°.
 */

#include <ESP32Servo.h>
#include <BluetoothSerial.h>
#include <Preferences.h>

#if !defined(CONFIG_BT_SPP_ENABLED)
#error "Bluetooth serial indisponivel nesta placa. Selecione um ESP32 classico (nao S2/S3/C3/C6/H2)."
#endif

const char *NOME_BT = "BCI_Ortese";   // igual a config.SERVO_BT_NOME
const int SERVO_PIN = 18;

Servo servo;
BluetoothSerial SerialBT;
Preferences prefs;

// Estado ajustável pelo console. Nada é const de propósito.
int repouso  = 140;
int angle    = repouso;
int passo    = 3;
int minAng   = 0;
int maxAng   = 180;
int pulsoMin = 500;    // ponytail: calibração do S3003; alargue com cuidado, batente mecânico
int pulsoMax = 2500;
bool tinhaClienteBT = false;

String linha;          // acumula um comando de console (letra + número)

void mover(int novo) {
  angle = constrain(novo, minAng, maxAng);
  servo.write(angle);
}

void estado(Stream &c) {
  c.printf("angle=%d passo=%d repouso=%d min=%d max=%d pulso=%d..%dus\r\n",
           angle, passo, repouso, minAng, maxAng, pulsoMin, pulsoMax);
}

// Comando de uma letra: executa na hora e ecoa "angle=<n>" (o Python conta esses ecos).
bool umByte(char c, Stream &canal) {
  if      (c == 'L') mover(angle - passo);
  else if (c == 'R') mover(angle + passo);
  else if (c == 'C') mover(repouso);
  else if (c >= '0' && c <= '9') { passo = max(1, c - '0'); canal.printf("Novo passo: %d\r\n", passo); }
  else if (c == '?') { estado(canal); return true; }
  else return false;
  canal.printf("angle=%d\r\n", angle);
  return true;
}

// Grava a cada H/N/X/U, ou seja, uma vez por conexão do PC. A flash aguenta isso folgado.
void salvar() {
  prefs.putInt("rep", repouso); prefs.putInt("min", minAng); prefs.putInt("max", maxAng);
  prefs.putInt("pmin", pulsoMin); prefs.putInt("pmax", pulsoMax);
}

// Comando de console completo, já sem o Enter.
void linhaCompleta(const String &s, Stream &canal) {
  char op = s[0];
  int n = s.substring(1).toInt();
  switch (op) {
    case 'A': mover(n); break;
    case 'P': passo = max(1, n); break;
    case 'H': repouso = constrain(n, 0, 180); break;
    case 'N': minAng = constrain(n, 0, 180); if (angle < minAng) mover(angle); break;
    case 'X': maxAng = constrain(n, 0, 180); if (angle > maxAng) mover(angle); break;
    case 'U': {
      int v = s.indexOf(',');
      if (v < 0) { canal.println("uso: U<min>,<max>"); return; }
      int a = s.substring(1, v).toInt(), b = s.substring(v + 1).toInt();
      if (a < 300 || b > 3000 || a >= b) { canal.println("pulsos fora de 300..3000 ou min>=max"); return; }
      pulsoMin = a; pulsoMax = b;
      servo.detach(); servo.attach(SERVO_PIN, pulsoMin, pulsoMax); servo.write(angle);
      break;
    }
    default: canal.printf("comando desconhecido: %s\r\n", s.c_str()); return;
  }
  estado(canal);
}

void processar(char c, Stream &canal) {
  if (c == '\r' || c == '\n') {               // fim de um comando de console
    if (linha.length()) linhaCompleta(linha, canal);
    linha = "";
    return;
  }
  if (linha.length() == 0 && umByte(c, canal)) return;   // L/R/C/dígito/? soltos
  linha += c;
  if (linha.length() > 16) linha = "";        // lixo: descarta
}

void setup() {
  Serial.begin(115200);
  delay(200);
  prefs.begin("ortese", false);
  repouso  = prefs.getInt("rep", repouso);
  minAng   = prefs.getInt("min", minAng);
  maxAng   = prefs.getInt("max", maxAng);
  pulsoMin = prefs.getInt("pmin", pulsoMin);
  pulsoMax = prefs.getInt("pmax", pulsoMax);
  angle    = constrain(repouso, minAng, maxAng);
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, pulsoMin, pulsoMax);
  servo.write(angle);

  if (!SerialBT.begin(NOME_BT)) Serial.println("ERRO: Bluetooth nao iniciou. So USB.");

  Serial.println("Firmware LIVRE. L/R/C/0-9 | A<n> P<n> H<n> N<n> X<n> U<a>,<b> ? (+Enter)");
  Serial.print("Bluetooth: "); Serial.println(NOME_BT);
  estado(Serial);
}

void loop() {
  if (Serial.available())   processar(Serial.read(), Serial);
  if (SerialBT.available()) processar(SerialBT.read(), SerialBT);

  bool temClienteBT = SerialBT.hasClient();
  if (tinhaClienteBT && !temClienteBT) {
    mover(repouso);
    Serial.println("Bluetooth desconectado -> repouso");
  }
  tinhaClienteBT = temClienteBT;
}
