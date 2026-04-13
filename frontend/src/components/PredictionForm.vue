<script setup>
import { usePredictionForm } from "../features/prediction/usePredictionForm";

const { error, fields, form, loading, result, submitPrediction } = usePredictionForm();
</script>

<template>
  <form class="prediction-form" @submit.prevent="submitPrediction">
    <div class="field-grid">
      <label v-for="field in fields" :key="field.name" class="field">
        <span>{{ field.label }}</span>
        <input
          v-model="form[field.name]"
          :name="field.name"
          inputmode="decimal"
          required
          step="any"
          type="number"
        />
      </label>
    </div>

    <button type="submit" :disabled="loading">
      {{ loading ? "Predicting..." : "Predict winner" }}
    </button>

    <p v-if="result" class="result">{{ result }}</p>
    <p v-if="error" class="error">{{ error }}</p>
  </form>
</template>

<style scoped>
.prediction-form {
  background: #ffffff;
  border: 1px solid #dce4df;
  border-radius: 8px;
  box-shadow: 0 12px 32px rgba(23, 32, 26, 0.08);
  padding: 24px;
}

.field-grid {
  display: grid;
  gap: 18px;
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.field {
  display: grid;
  gap: 8px;
}

.field span {
  color: #384843;
  font-weight: 700;
}

input {
  border: 1px solid #b7c5bf;
  border-radius: 8px;
  min-height: 44px;
  padding: 8px 10px;
}

input:focus {
  border-color: #31706b;
  outline: 3px solid rgba(49, 112, 107, 0.18);
}

button {
  background: #31706b;
  border: 0;
  border-radius: 8px;
  color: #ffffff;
  cursor: pointer;
  font-weight: 800;
  margin-top: 24px;
  min-height: 46px;
  padding: 0 18px;
}

button:disabled {
  cursor: wait;
  opacity: 0.72;
}

.result,
.error {
  border-radius: 8px;
  font-weight: 800;
  margin: 20px 0 0;
  padding: 14px;
}

.result {
  background: #e3f3ed;
  color: #20564f;
}

.error {
  background: #fde7e7;
  color: #9f2c2c;
}

@media (max-width: 720px) {
  .field-grid {
    grid-template-columns: 1fr;
  }

  .prediction-form {
    padding: 18px;
  }
}
</style>
