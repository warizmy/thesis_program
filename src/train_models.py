import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from imblearn.over_sampling import SMOTE  # Import SMOTE untuk oversampling

# Load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Preprocess data
def preprocess_data(data):
    data = data.dropna()
    columns_to_drop = ['nama', 'pemrograman_web', 'data_mining', 'pengolahan_citra', 'kecerdasarn_buatan', 'struktur_data', 'sistem_cerdas', 'machine_learning']
    data = data.drop(columns=columns_to_drop)
    return data

# Create target column based on criteria
def create_target_column(data):
    data['program_akselerasi'] = "Tidak"
    data.loc[
        (data['total_sks'] >= 115) & 
        (data['ipk'] >= 3.6), 
        'program_akselerasi'
    ] = "Ya"
    return data

# EDA function
def perform_eda(data):
    print("Distribusi Kelas Target:")
    print(data['program_akselerasi'].value_counts())
    
    # Visualisasi distribusi kelas target
    plt.figure(figsize=(6, 4))
    sns.countplot(x='program_akselerasi', data=data)
    plt.title('Distribusi Kelas Target: program_akselerasi')
    plt.show()

    # Visualisasi distribusi fitur numerik
    numerical_features = ['ipk', 'total_sks']  # Daftar fitur numerik
    data[numerical_features].hist(bins=20, figsize=(10, 6))
    plt.suptitle('Distribusi Fitur Numerik')
    plt.show()

    # Memeriksa nilai yang hilang
    print("\nMissing Values:")
    print(data.isnull().sum())

    # Korelasi antar fitur numerik
    correlation_matrix = data[numerical_features].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Heatmap Korelasi Fitur Numerik')
    plt.show()

# Oversampling data untuk kelas minoritas
def oversample_data(X, y):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Train models using k-fold cross-validation with parameter search
def train_models(X, y, n_splits=5):
    models = {
        'Naive Bayes': GaussianNB(),
        'KNN': KNeighborsClassifier()
    }

    # Parameter grid for Naive Bayes and KNN
    param_grid = {
        'Naive Bayes': {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]},
        'KNN': {
            'n_neighbors': [3, 5, 7, 9, 11, 13],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    }

    results = {name: {'train_accuracy': [], 'val_accuracy': []} for name in models.keys()}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_params = {}

    for name, model in models.items():
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid[name], cv=kf, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_

        # Store best parameters
        best_params[name] = grid_search.best_params_

        # Calculate and store train and test accuracy
        train_accuracy = grid_search.best_score_
        test_predictions = best_model.predict(X)
        test_accuracy = accuracy_score(y, test_predictions)

        # Save the results for plotting and table
        results[name]['train_accuracy'].append(train_accuracy)
        results[name]['val_accuracy'].append(test_accuracy)

    return results, best_params

# Save the best model
def save_model(model, model_name):
    joblib.dump(model, f'./models/{model_name}.joblib')

# Plot learning curve
def plot_learning_curve(model, X, y, title="Learning Curve", ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(0.1, 1.0, 10)):
    plt.figure(figsize=(10, 6))
    train_sizes, train_scores, valid_scores = learning_curve(model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    # Menghitung rata-rata dan deviasi standar untuk score
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    valid_mean = valid_scores.mean(axis=1)
    valid_std = valid_scores.std(axis=1)

    # Plot learning curve
    plt.plot(train_sizes, train_mean, label='Train Accuracy', color='blue', marker='o')
    plt.plot(train_sizes, valid_mean, label='Validation Accuracy', color='green', marker='x')
    
    # Plot area error
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color='blue', alpha=0.2)
    plt.fill_between(train_sizes, valid_mean - valid_std, valid_mean + valid_std, color='green', alpha=0.2)

    plt.title(title)
    plt.xlabel('Training Size')
    plt.ylabel('Accuracy')
    if ylim:
        plt.ylim(*ylim)
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load raw data
    raw_data_path = './data/raw/data.csv'
    data = load_data(raw_data_path)

    # Preprocess data
    processed_data = preprocess_data(data)
    processed_data = create_target_column(processed_data)

    # Perform EDA
    perform_eda(processed_data)

    # Save preprocessed data
    processed_data.to_csv('./data/preprocessed/data_preprocessed.csv', index=False)

    # Split data into features and target
    X = processed_data.drop('program_akselerasi', axis=1)
    y = processed_data['program_akselerasi']

    # Normalize features
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Oversample data
    X_resampled, y_resampled = oversample_data(X, y)

    # Periksa distribusi kelas setelah oversampling
    print("Distribusi kelas setelah oversampling:")
    print(pd.Series(y_resampled).value_counts())

    # Train models with k-fold cross-validation (5 fold)
    results, best_params = train_models(X_resampled, y_resampled, n_splits=5)

    # Print the best parameters for each model
    print("Best Parameters for Each Model:")
    for model_name, params in best_params.items():
        print(f"{model_name}: {params}")

    # Creating a DataFrame to display the train and test accuracies
    accuracies = []
    for model_name in results:
        accuracies.append({
            'Model': model_name,
            'Best Train Accuracy': results[model_name]['train_accuracy'][0],
            'Best Validation Accuracy': results[model_name]['val_accuracy'][0]
        })
    
    accuracy_df = pd.DataFrame(accuracies)
    print("\nTrain and Validation Accuracy Comparison:")
    print(accuracy_df)

    # Plot comparison of train and test accuracy
    model_names = list(results.keys())
    train_accuracies = [results[name]['train_accuracy'][0] for name in model_names]
    test_accuracies = [results[name]['val_accuracy'][0] for name in model_names]

    plt.figure(figsize=(8, 5))
    bar_width = 0.35
    index = range(len(model_names))

    plt.bar(index, train_accuracies, bar_width, label='Train Accuracy', alpha=0.7)
    plt.bar([i + bar_width for i in index], test_accuracies, bar_width, label='Test Accuracy', alpha=0.7)

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy for Naive Bayes and KNN')
    plt.xticks([i + bar_width / 2 for i in index], model_names)
    plt.legend()
    plt.show()

    # Plot learning curves for both models
    print("\nPlotting Learning Curves:")
    plot_learning_curve(GaussianNB(), X_resampled, y_resampled, title="Learning Curve Naive Bayes")
    plot_learning_curve(KNeighborsClassifier(n_neighbors=5), X_resampled, y_resampled, title="Learning Curve KNN")

    # Final Test Evaluation
    print('Final Test Evaluation:')
    model_performance = {}
    for model_name, model in {'Naive Bayes': GaussianNB(), 'KNN': KNeighborsClassifier()}.items():
        model.fit(X_resampled, y_resampled)
        test_predictions = model.predict(X_resampled)
        test_accuracy = accuracy_score(y_resampled, test_predictions)
        test_precision = precision_score(y_resampled, test_predictions, pos_label="Ya")
        test_recall = recall_score(y_resampled, test_predictions, pos_label="Ya")
        test_f1 = f1_score(y_resampled, test_predictions, pos_label="Ya")
        
        model_performance[model_name] = {
            'accuracy': test_accuracy,
            'precision': test_precision,
            'recall': test_recall,
            'f1': test_f1
        }

        print(f'\nModel: {model_name}')
        print(f'Accuracy: {test_accuracy:.4f}')
        print(f'Precision: {test_precision:.4f}')
        print(f'Recall: {test_recall:.4f}')
        print(f'F1 Score: {test_f1:.4f}')
    
    # Select the best model based on F1 Score or Accuracy
    best_model_name = max(model_performance, key=lambda x: model_performance[x]['accuracy'])
    best_model = GaussianNB() if best_model_name == 'Naive Bayes' else KNeighborsClassifier()
    best_model.fit(X_resampled, y_resampled)

    # Save the best model
    save_model(best_model, best_model_name.lower().replace(' ', '_'))

    print(f'\nBest model saved: {best_model_name}')
