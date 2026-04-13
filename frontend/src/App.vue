<script setup>
import PredictionForm from "./components/PredictionForm.vue";
</script>

<template>
  <main class="page-shell">
    <section class="intro">
      <p class="eyebrow">League of Legends Diamond Ranked Games</p>
      <h1>10-minute win predictor</h1>
      <p>
        Enter the early-game match stats and estimate whether blue side or red side is favored.
      </p>
    </section>

    <PredictionForm />
  </main>
</template>

<style>
:root {
  color: #17201a;
  background: #f6f8f4;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-synthesis: none;
  text-rendering: optimizeLegibility;
}

body {
  margin: 0;
  min-width: 320px;
}

button,
input {
  font: inherit;
}

.page-shell {
  width: min(1040px, calc(100% - 32px));
  margin: 0 auto;
  padding: 48px 0;
}

.intro {
  margin-bottom: 32px;
}

.eyebrow {
  color: #31706b;
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0;
  margin: 0 0 8px;
  text-transform: uppercase;
}

h1 {
  font-size: 2.8rem;
  line-height: 1.05;
  margin: 0 0 12px;
}

.intro p:last-child {
  color: #51615c;
  font-size: 1.05rem;
  max-width: 680px;
  margin: 0;
}
</style>
