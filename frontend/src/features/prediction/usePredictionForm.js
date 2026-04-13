import { reactive, ref } from "vue";

import { predictMatch } from "../../services/predictionApi";
import { predictionFields } from "./predictionFields";

function createInitialFormState() {
  return Object.fromEntries(predictionFields.map((field) => [field.name, 0]));
}

function buildPredictionPayload(form) {
  return Object.fromEntries(
    predictionFields.map((field) => [field.name, Number(form[field.name])])
  );
}

function isValidPayload(payload) {
  return Object.values(payload).every((value) => Number.isFinite(value));
}

export function usePredictionForm() {
  const form = reactive(createInitialFormState());
  const result = ref("");
  const error = ref("");
  const loading = ref(false);

  async function submitPrediction() {
    const payload = buildPredictionPayload(form);
    result.value = "";
    error.value = "";

    if (!isValidPayload(payload)) {
      error.value = "All fields must contain numeric values.";
      return;
    }

    loading.value = true;
    try {
      const response = await predictMatch(payload);
      result.value = response.result;
    } catch (requestError) {
      error.value = requestError.message;
    } finally {
      loading.value = false;
    }
  }

  return {
    error,
    fields: predictionFields,
    form,
    loading,
    result,
    submitPrediction
  };
}
