#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define MAX_ROWS 10000
#define MAX_COLS 50

double X[MAX_ROWS][MAX_COLS];
double y[MAX_ROWS];
double theta[MAX_COLS + 1];
int rows = 0;

int compare(const void *a, const void *b) {
    double diff = *(double*)a - *(double*)b;
    return (diff > 0) - (diff < 0);
}

void load_data(const char* xfile, const char* yfile, int cols) {
    FILE* xf = fopen(xfile, "r");
    FILE* yf = fopen(yfile, "r");

    while (fscanf(yf, "%lf", &y[rows]) != EOF) {
        for (int j = 0; j < cols; j++) {
            fscanf(xf, "%lf", &X[rows][j]);
        }
        rows++;
    }

    fclose(xf);
    fclose(yf);
}

void compute_statistics() {
    double mean = 0.0, std_dev = 0.0, min_val = y[0], max_val = y[0];
    
    #pragma omp parallel for reduction(+:mean) reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < rows; i++) {
        mean += y[i];
        if (y[i] < min_val) min_val = y[i];
        if (y[i] > max_val) max_val = y[i];
    }

    mean /= rows;

    #pragma omp parallel for reduction(+:std_dev)
    for (int i = 0; i < rows; i++) {
        std_dev += (y[i] - mean) * (y[i] - mean);
    }
    std_dev = sqrt(std_dev / rows);

    double temp[MAX_ROWS];
    for (int i = 0; i < rows; i++) temp[i] = y[i];

    qsort(temp, rows, sizeof(double), compare);
    double median = (rows % 2 == 0) ? (temp[rows/2 - 1] + temp[rows/2]) / 2.0 : temp[rows/2];

    printf("📊 Target Column (y) Statistics (computed in C with OpenMP):\n");
    printf("Mean: %.2f\n", mean);
    printf("Standard Deviation: %.2f\n", std_dev);
    printf("Min: %.2f\n", min_val);
    printf("Max: %.2f\n", max_val);
    printf("Median: %.2f\n", median);
}

void compute_feature_scaling(int cols) {
    printf("\n📐 Feature Scaling (Standardization Stats):\n");
    
    for (int j = 0; j < cols; j++) {
        double sum = 0.0, std_dev = 0.0;

        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < rows; i++) {
            sum += X[i][j];
        }

        double mean = sum / rows;

        #pragma omp parallel for reduction(+:std_dev)
        for (int i = 0; i < rows; i++) {
            std_dev += (X[i][j] - mean) * (X[i][j] - mean);
        }

        std_dev = sqrt(std_dev / rows);

        printf("Feature %d ➜ Mean: %.4f | Std Dev: %.4f\n", j + 1, mean, std_dev);
    }
}

void gradient_descent(int cols, int epochs, double lr) {
    FILE* log = fopen("cost_log.csv", "w");
    fprintf(log, "epoch,cost\n");

    for (int e = 0; e < epochs; e++) {
        double grad[MAX_COLS + 1] = {0};
        double cost = 0;

        #pragma omp parallel for reduction(+:cost)
        for (int i = 0; i < rows; i++) {
            double pred = theta[0];
            for (int j = 0; j < cols; j++)
                pred += theta[j+1] * X[i][j];

            double err = pred - y[i];
            cost += err * err;

            #pragma omp critical
            {
                grad[0] += err;
                for (int j = 0; j < cols; j++)
                    grad[j+1] += err * X[i][j];
            }
        }

        for (int j = 0; j <= cols; j++)
            theta[j] -= lr * grad[j] / rows;

        if (e % 10 == 0)
            fprintf(log, "%d,%.6f\n", e, cost / (2 * rows));
    }

    fclose(log);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        printf("Usage: ./gd_omp <num_features> <epochs> <lr>\n");
        return 1;
    }

    int cols = atoi(argv[1]);
    int epochs = atoi(argv[2]);
    double lr = atof(argv[3]);

    double start_time = omp_get_wtime();

    load_data("X.csv", "y.csv", cols);
    compute_statistics();
    //gradient_descent(cols, epochs, lr);

    double end_time = omp_get_wtime();

    printf("\nExecution Time: %.6f seconds\n", end_time - start_time);

    return 0;
}
