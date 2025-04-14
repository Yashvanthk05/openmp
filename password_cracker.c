#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <math.h>
#include <stdbool.h>
#include <zip.h>

#define CHARSET "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
#define CHARSET_LEN 62
#define MAX_PASSWORD_LENGTH 20

static volatile bool found = false;
static char cracked_password[MAX_PASSWORD_LENGTH + 1] = {0};
static long processed = 0;

void calculate_password(char *password, long index, int length, const char *charset) {
    for (int i = 0; i < length; i++) {
        password[length - i - 1] = charset[index % CHARSET_LEN];
        index /= CHARSET_LEN;
    }
    password[length] = '\0';
}

bool test_password(const char *password, const char *filename) {
    int err = 0;
    zip_t *za = zip_open(filename, ZIP_RDONLY, &err);
    if (!za) return false;
    zip_set_default_password(za, password);
    zip_int64_t num_entries = zip_get_num_entries(za, 0);
    if (num_entries == 0) {
        zip_close(za);
        return false;
    }
    for (zip_uint64_t i = 0; i < num_entries; i++) {
        zip_stat_t sb;
        if (zip_stat_index(za, i, 0, &sb) == -1) {
            zip_close(za);
            return false;
        }
        zip_file_t *zf = zip_fopen_index(za, i, 0);
        if (!zf) {
            zip_close(za);
            return false;
        }
        char *buffer = malloc(sb.size);
        if (!buffer) {
            zip_fclose(zf);
            zip_close(za);
            return false;
        }
        zip_int64_t bytes_read = zip_fread(zf, buffer, sb.size);
        free(buffer);
        zip_fclose(zf);
        if (bytes_read != sb.size) {
            zip_close(za);
            return false;
        }
    }
    zip_close(za);
    return true;
}

void generate_and_test(int length, const char *filename) {
    long total = pow(CHARSET_LEN, length);
    printf("🔍 Searching through %ld combinations (length=%d)...\n", total, length);
    #pragma omp parallel
    {
        char guess[MAX_PASSWORD_LENGTH + 1];
        const char *charset = CHARSET;
        #pragma omp for schedule(dynamic, 1000)
        for (long i = 0; i < total; ++i) {
            if (found) continue;
            calculate_password(guess, i, length, charset);
            if (test_password(guess, filename)) {
                #pragma omp critical
                {
                    if (!found) {
                        found = true;
                        strcpy(cracked_password, guess);
                        printf("\n✅ Cracked by thread %d: %s\n", omp_get_thread_num(), cracked_password);
                    }
                }
            }
            #pragma omp atomic
            processed++;
            if (omp_get_thread_num() == 0 && processed % 100000 == 0) {
                printf("\rProgress: %.2f%%", 100.0 * processed / total);
                fflush(stdout);
            }
        }
    }
    if (!found) {
        printf("\n❌ Password not found for length %d\n", length);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <length> <file.zip>\n", argv[0]);
        return 1;
    }
    int length = atoi(argv[1]);
    const char *filename = argv[2];
    if (length <= 0 || length > MAX_PASSWORD_LENGTH) {
        printf("Password length must be between 1 and %d\n", MAX_PASSWORD_LENGTH);
        return 1;
    }
    printf("🧵 Using %d threads\n", omp_get_max_threads());
    double start = omp_get_wtime();
    generate_and_test(length, filename);
    double end = omp_get_wtime();
    if (found) {
        printf("✅ Password: %s\n", cracked_password);
    }
    printf("⏱️ Time taken: %.2f seconds\n", end - start);
    return 0;
}
