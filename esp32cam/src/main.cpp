/*
 * Cavalo Mola — firmware ESP32-CAM (AI-Thinker), sem câmera.
 *
 * Lê do microSD (modo SD_MMC 1 bit: CLK=14, CMD=15, D0=2 — não usa o LED flash do GPIO4):
 *   /pav/entrada/lista.txt                nomes das amostras (um por linha)
 *   /pav/entrada/<nome>.pgm               intensidade 512x320 (P5)
 *   /pav/entrada/<nome>_roi.pgm           região da pista (0/255)
 * Escreve:
 *   /pav/saida/resultados.csv             quantificação + hash das máscaras + tempo (us)
 *   /pav/saida/<nome>_res.pgm             máscara de saída (128 trinca, 200 esqueleto, 255 buraco)
 *
 * O processamento é o mesmo nucleo.c verificado bit a bit contra o Python
 * (scripts/comparar_esp32.py).
 */
#include <Arduino.h>
#include "FS.h"
#include "SD_MMC.h"
#include "esp_timer.h"
#include "nucleo.h"
#include "estrada.h"
#include "params.h"
#include "formato_csv.h"

static const char *DIR_ENT = "/pav/entrada";
static const char *DIR_SAI = "/pav/saida";
static const bool GRAVAR_MASCARAS = true;
static const int LED_STATUS = 33;   // LED vermelho da placa (ativo em nível baixo)
static uint32_t relogio_us() { return (uint32_t)esp_timer_get_time(); }
static const char *ETAPAS[5] = {"estatisticas", "buracos", "trincas", "esqueleto", "medidas"};

static bool lerPGM(const String &caminho, uint8_t *buf, size_t cap, int &w, int &h) {
    File f = SD_MMC.open(caminho, FILE_READ);
    if (!f) return false;
    // cabeçalho: "P5" w h maxval, separados por espaço em branco
    String tok[4]; int nt = 0; String atual;
    while (nt < 4 && f.available()) {
        char c = (char)f.read();
        if (isspace((unsigned char)c)) { if (atual.length()) { tok[nt++] = atual; atual = ""; } }
        else atual += c;
    }
    if (nt < 4 || tok[0] != "P5") { f.close(); return false; }
    w = tok[1].toInt(); h = tok[2].toInt();
    size_t n = (size_t)w * h;
    if (n > cap) { f.close(); return false; }
    size_t lidos = f.read(buf, n);
    f.close();
    return lidos == n;
}

static void gravarPGM(const String &caminho, const uint8_t *buf, int w, int h) {
    File f = SD_MMC.open(caminho, FILE_WRITE);
    if (!f) return;
    f.printf("P5\n%d %d\n255\n", w, h);
    f.write(buf, (size_t)w * h);
    f.close();
}

// Estado final, sinalizado pelo LED em loop() até reiniciar a placa:
//   0 = processando (LED apagado), 1 = concluído (pisca lento, 500 ms), 2 = erro (pisca rápido, 100 ms)
static volatile int estado = 0;

void setup() {
    pinMode(LED_STATUS, OUTPUT);
    digitalWrite(LED_STATUS, HIGH);
    Serial.begin(115200);
    delay(500);
    Serial.println("\n=== Cavalo Mola — deteccao e quantificacao de danos em pavimentos ===");

    if (!psramFound()) { Serial.println("ERRO: PSRAM nao encontrada"); estado = 2; return; }
    if (!SD_MMC.begin("/sdcard", true)) {       // true = modo 1 bit
        Serial.println("ERRO: falha ao montar o cartao SD"); estado = 2; return;
    }
    Serial.printf("SD: %llu MB | PSRAM livre: %u KB\n", SD_MMC.cardSize() / (1024 * 1024),
                  (unsigned)(ESP.getFreePsram() / 1024));

    const int W = 512, H = 320;
    const size_t N = (size_t)W * H;
    uint8_t *g = (uint8_t *)ps_malloc(N), *roi = (uint8_t *)ps_malloc(N), *saida = (uint8_t *)ps_malloc(N);
    void *trab = ps_malloc(pav_bytes_trabalho(W, H));
    if (!g || !roi || !saida || !trab) { Serial.println("ERRO: memoria insuficiente"); estado = 2; return; }

    File lista = SD_MMC.open(String(DIR_ENT) + "/lista.txt", FILE_READ);
    if (!lista) { Serial.println("ERRO: /pav/entrada/lista.txt nao encontrado"); estado = 2; return; }
    SD_MMC.mkdir("/pav");
    SD_MMC.mkdir(DIR_SAI);
    const String CSV = String(DIR_SAI) + "/resultados.csv";
    { File c = SD_MMC.open(CSV, FILE_WRITE); c.println(PAV_CSV_CABECALHO); c.close(); }

    pav_params_t p = PAV_PARAMS_PADRAO;
    pav_definir_relogio(relogio_us);
    pav_estrada_params_t pe = PAV_ESTRADA_PADRAO;
    uint64_t soma_etapa[5] = {0, 0, 0, 0, 0};
    int total = 0; int64_t soma_us = 0, max_us = 0;
    while (lista.available()) {
        String nome = lista.readStringUntil('\n'); nome.trim();
        if (!nome.length()) continue;
        int w, h, w2 = 0, h2 = 0;
        if (!lerPGM(String(DIR_ENT) + "/" + nome + ".pgm", g, N, w, h)) {
            Serial.printf("%s: falha de leitura\n", nome.c_str());
            continue;
        }
        // Área de análise: máscara da pista (se existir) OU estrada automática pela saturação
        bool tem_roi = lerPGM(String(DIR_ENT) + "/" + nome + "_roi.pgm", roi, N, w2, h2) && w2 == w && h2 == h;
        int64_t t0 = esp_timer_get_time();
        uint32_t h_estrada = 0, t_estrada = 0;
        if (tem_roi) {
            for (size_t i = 0; i < (size_t)w * h; i++) roi[i] = roi[i] ? 1 : 0;
        } else {
            // a saturação é lida no buffer 'saida' (que só é preenchido depois)
            if (!lerPGM(String(DIR_ENT) + "/" + nome + "_sat.pgm", saida, N, w2, h2) || w2 != w || h2 != h) {
                Serial.printf("%s: sem _roi.pgm nem _sat.pgm\n", nome.c_str());
                continue;
            }
            t0 = esp_timer_get_time();   // não conta a leitura do SD
            pav_estrada(g, saida, w, h, &pe, trab, roi);
            h_estrada = pav_fnv1a(roi, (size_t)w * h);
            t_estrada = (uint32_t)(esp_timer_get_time() - t0);
        }
        pav_quant_t q;
        int rc = pav_analisar(g, roi, w, h, &p, trab, &q, GRAVAR_MASCARAS ? saida : NULL);
        int64_t us = esp_timer_get_time() - t0;
        q.hash_estrada = h_estrada; q.t_estrada_us = t_estrada;
        char linha[384];
        pav_csv_linha(linha, sizeof linha, nome.c_str(), rc, &q, (long)us);
        // abre/fecha a cada linha: o FAT só atualiza o tamanho do arquivo no close,
        // então assim nada se perde se a placa for desligada no meio
        { File c = SD_MMC.open(CSV, FILE_APPEND); c.println(linha); c.close(); }
        if (GRAVAR_MASCARAS && rc == 0) gravarPGM(String(DIR_SAI) + "/" + nome + "_res.pgm", saida, w, h);
        Serial.printf("%-10s %4ld ms | trincas %3ld (%5ld px) | buracos %2ld (%4ld.%02ld%%) | indice %4.1f %s\n",
                      nome.c_str(), (long)(us / 1000), (long)q.n_trincas, (long)q.comprimento_trincas_px,
                      (long)q.n_buracos, (long)(q.area_buracos_pct_100 / 100), (long)(q.area_buracos_pct_100 % 100),
                      q.indice_dano_1000 / 10.0, PAV_SEVERIDADE[q.severidade]);
        if (!tem_roi) Serial.printf("           estrada automatica: %.0f ms\n", t_estrada / 1000.0);
        Serial.printf("           etapas (ms): estat %.0f | buracos %.0f | trincas %.0f | esqueleto %.0f | medidas %.0f\n",
                      q.t_etapa_us[0] / 1000.0, q.t_etapa_us[1] / 1000.0, q.t_etapa_us[2] / 1000.0,
                      q.t_etapa_us[3] / 1000.0, q.t_etapa_us[4] / 1000.0);
        for (int e = 0; e < 5; e++) soma_etapa[e] += q.t_etapa_us[e];
        total++; soma_us += us; if (us > max_us) max_us = us;
    }
    lista.close();
    Serial.printf("Concluido: %d imagens | media %.1f ms | max %.1f ms\n", total,
                  total ? soma_us / 1000.0 / total : 0.0, max_us / 1000.0);
    { File r = SD_MMC.open(String(DIR_SAI) + "/resumo.txt", FILE_WRITE);
      r.printf("imagens=%d\nmedia_ms=%.1f\nmax_ms=%.1f\npsram_livre_kb=%u\ncpu_mhz=%u\n", total,
               total ? soma_us / 1000.0 / total : 0.0, max_us / 1000.0, (unsigned)(ESP.getFreePsram() / 1024),
               (unsigned)getCpuFrequencyMhz());
      for (int e = 0; e < 5; e++)
          r.printf("media_%s_ms=%.1f\n", ETAPAS[e], total ? soma_etapa[e] / 1000.0 / total : 0.0);
      r.close(); }
    Serial.println("Resultados em /pav/saida/resultados.csv");
    Serial.println("LED piscando = concluido. Pode retirar o cartao.");
    estado = 1;
}

void loop() {
    // Pisca indefinidamente até reiniciar: lento = concluído, rápido = erro
    if (estado == 0) { delay(50); return; }
    int meio_periodo = estado == 1 ? 500 : 100;
    digitalWrite(LED_STATUS, LOW);  delay(meio_periodo);   // LED ativo em nível baixo
    digitalWrite(LED_STATUS, HIGH); delay(meio_periodo);
}
