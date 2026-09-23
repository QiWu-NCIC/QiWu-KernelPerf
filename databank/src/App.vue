<template>
  <div class="app-shell">
    <header class="qiwu-brand-bar">
      <div class="qiwu-brand-inner">
        <img class="qiwu-logo" src="./assets/images/logo.png" alt="QiWu" />
      </div>
    </header>
    <nav class="operator-tabs" aria-label="Benchmark operator">
      <button :class="{ active: selectedOperator === 'spmv' }" @click="selectOperator('spmv')">SpMV</button>
      <button :class="{ active: selectedOperator === 'spmm' }" @click="selectOperator('spmm')">SpMM</button>
    </nav>
    <Spmv v-if="selectedOperator === 'spmv'" />
    <Spmm v-else />
  </div>
</template>

<script setup>
import { ref } from "vue";
import Spmv from "./views/SPMV/index.vue";
import Spmm from "./views/SPMM/index.vue";

const selectedOperator = ref(window.location.hash === "#spmm" ? "spmm" : "spmv");
function selectOperator(value) {
  selectedOperator.value = value;
  window.location.hash = value;
}
</script>

<style scoped>
.app-shell {
  min-height: 100vh;
}

.qiwu-brand-bar {
  width: 100%;
  padding: 20px 10%;
  background: #254d9e;
}

.qiwu-brand-inner {
  display: flex;
  align-items: center;
  justify-content: space-between;
  width: 100%;
}

.qiwu-logo {
  display: block;
  width: auto;
  height: 50px;
}

.operator-tabs {
  display: flex;
  gap: 8px;
  padding: 12px 10%;
  background: #17366f;
}

.operator-tabs button {
  border: 1px solid #9bb4e8;
  border-radius: 4px;
  padding: 7px 16px;
  color: white;
  background: transparent;
  cursor: pointer;
}

.operator-tabs button.active {
  color: #17366f;
  background: white;
}

@media (max-width: 750px) {
  .qiwu-brand-bar {
    padding: 10px 10%;
  }

  .qiwu-logo {
    height: 40px;
  }
}
</style>
