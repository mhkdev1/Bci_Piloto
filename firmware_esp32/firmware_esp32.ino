/*
 * firmware_esp32.ino — controle do servo da órtese, lado ESP32.
 *
 * Este é o código gravado DENTRO do ESP32. Do outro lado da conexão ficam:
 *   controle_servo_1.py    controle manual por teclado (setas)
 *   inferir_offline.py     inferência em dados gravados, com --ortese
 *   tempo_real.py          inferência ao vivo, com o capacete
 *   capturar_tempo_real.py inferência ao vivo, gravando o EEG
 *
 * Todos falam o mesmo protocolo. Nada aqui depende de qual deles está conectado.
 *
 * ---------------------------------------------------------------------------
 * CONEXÃO — Bluetooth ou cabo USB, com o mesmo protocolo
 * ---------------------------------------------------------------------------
 * O ESP32 aparece para o PC como o aparelho Bluetooth "BCI_Ortese" (perfil serial,
 * SPP). Depois de pareado no Windows ele ganha uma porta COM, e o PC acha essa porta
 * sozinho pelo nome — config.SERVO_BT_NOME tem que ser igual a NOME_BT, abaixo.
 * O cabo USB continua funcionando ao mesmo tempo: cada resposta volta pelo canal de
 * onde veio o comando.
 *
 * Só o ESP32 "clássico" (ESP32-WROOM/WROVER, DevKit V1) tem Bluetooth serial. S2, S3,
 * C3, C6 e H2 não têm, e o sketch para com um erro claro na compilação.
 *
 * ---------------------------------------------------------------------------
 * LIGAÇÃO
 * ---------------------------------------------------------------------------
 *   servo (sinal PWM)  ->  GPIO 18
 *   servo (VCC)        ->  fonte externa 5 V, NÃO o 5V do ESP32
 *   servo (GND)        ->  GND comum com o ESP32
 *
 * O GND precisa ser comum, senão o servo treme ou não responde. Alimentar o servo pelo
 * regulador da placa costuma derrubá-la no pico de corrente da partida.
 *
 * ---------------------------------------------------------------------------
 * PROTOCOLO — um byte por comando, sem terminador
 * ---------------------------------------------------------------------------
 *   'L'        angle = max(MIN_ANG, angle - stepDeg)
 *   'R'        angle = min(MAX_ANG, angle + stepDeg)
 *   'C'        angle = 90   (centro do curso; é a posição de repouso da órtese)
 *   '0'..'9'   stepDeg = dígito, com '0' virando 1
 *
 * A cada byte recebido o firmware imprime "angle=<n>". O lado do PC lê esse eco e o usa
 * como posição real, em vez de contar os pulsos às cegas — e precisa lê-lo de qualquer
 * jeito, senão a saída se acumula no buffer do host durante a sessão inteira.
 *
 * ---------------------------------------------------------------------------
 * ATENÇÃO AO CURSO
 * ---------------------------------------------------------------------------
 * MAX_ANG é 180: este firmware NÃO limita o curso à amplitude da órtese. Quem limita é
 * o PC (config.SERVO_AMPLITUDE, contado a partir do repouso). Isso funciona enquanto o
 * PC for quem comanda, mas não protege contra um terminal aberto por engano nem contra
 * o programa travar com a mão fechada.
 *
 * O que o firmware garante sozinho: se o cliente Bluetooth desconectar, a órtese volta ao
 * repouso. Um link sem fio cai sem aviso, e o PC não consegue mandar o 'C' por um link que
 * já caiu. O ESP32 percebe na hora quando o programa fecha a porta, mas só em até uns 20 s
 * quando o link some (PC fora de alcance, Bluetooth do PC desligado). Isso não cobre o
 * cabo USB, nem um programa que trave com a conexão aberta — para esses, a variante segura.
 *
 * Para uso com a órtese vestida, considere a variante em firmware_esp32_seguro/, que
 * acrescenta batente por software e retorno automático ao repouso. Ver o README.
 *
 * Dependência: biblioteca "ESP32Servo", pelo Gerenciador de Bibliotecas da IDE Arduino.
 * A BluetoothSerial já vem junto com o pacote de placas esp32.
 */

#include <ESP32Servo.h>
#include <BluetoothSerial.h>

#if !defined(CONFIG_BT_SPP_ENABLED)
#error "Bluetooth serial indisponivel nesta placa. Selecione um ESP32 classico (nao S2/S3/C3/C6/H2)."
#endif

const char *NOME_BT = "BCI_Ortese";   // nome no Bluetooth; igual a config.SERVO_BT_NOME

Servo servo;
BluetoothSerial SerialBT;

const int SERVO_PIN = 18;   // pino do sinal do servo
const int REPOUSO = 90;     // centro do curso; é para onde 'C' leva
int angle = REPOUSO;        // posição inicial (graus)
int stepDeg = 3;            // incremento por “passo”
const int MIN_ANG = -30;
const int MAX_ANG = 180;
bool tinhaClienteBT = false;

// Executa um byte do protocolo e responde pelo mesmo canal de onde ele veio.
void processar(char c, Stream &canal) {
  if (c == 'L') {                      // seta ESQUERDA
    angle = max(MIN_ANG, angle - stepDeg);
    servo.write(angle);
  } else if (c == 'R') {               // seta DIREITA
    angle = min(MAX_ANG, angle + stepDeg);
    servo.write(angle);
  } else if (c == 'C') {               // seta CIMA (centraliza)
    angle = REPOUSO;
    servo.write(angle);
  } else if (c >= '0' && c <= '9') {   // ajustar passo (0=1°, 9=9°)
    stepDeg = (c - '0');
    if (stepDeg == 0) stepDeg = 1;
    if (stepDeg > 10) stepDeg = 10;
    canal.printf("Novo passo: %d\r\n", stepDeg);
  }

  // feedback no mesmo canal, numa escrita só: no Bluetooth cada escrita vira um pacote, e
  // o eco em três pedaços triplica o que um link congestionado tem que escoar
  canal.printf("angle=%d\r\n", angle);
}

void setup() {
  Serial.begin(115200);
  delay(200);

  // S3003 trabalha a ~50 Hz; limites de pulso típicos 500–2500 µs
  servo.setPeriodHertz(50);
  servo.attach(SERVO_PIN, 500, 2500);
  servo.write(angle);

  if (!SerialBT.begin(NOME_BT)) {
    Serial.println("ERRO: o Bluetooth nao iniciou. So o cabo USB vai funcionar.");
  }

  Serial.println("Pronto. Comandos: L=esq, R=dir, C=centro, [0-9] muda passo.");
  Serial.print("Bluetooth: "); Serial.println(NOME_BT);
  Serial.print("Angulo: "); Serial.println(angle);
}

void loop() {
  if (Serial.available()) processar(Serial.read(), Serial);
  if (SerialBT.available()) processar(SerialBT.read(), SerialBT);

  // Perdeu o cliente Bluetooth: volta ao repouso, porque o PC já não tem como mandar 'C'.
  bool temClienteBT = SerialBT.hasClient();
  if (tinhaClienteBT && !temClienteBT) {
    angle = REPOUSO;
    servo.write(angle);
    Serial.println("Bluetooth desconectado -> repouso");
  }
  tinhaClienteBT = temClienteBT;
}
