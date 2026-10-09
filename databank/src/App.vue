<template>
  <div class="app-shell">
    <header class="qiwu-brand-bar">
      <div class="qiwu-brand-inner">
        <img class="qiwu-logo" src="./assets/images/logo.png" alt="QiWu" />
      </div>
    </header>
    <Spmv v-if="selectedOperator === 'spmv'" />
    <Spmm v-else />
  </div>
</template>

<script setup>
import { onBeforeUnmount, onMounted, ref } from "vue";
import Spmv from "./views/SPMV/index.vue";
import Spmm from "./views/SPMM/index.vue";

const selectedOperator = ref(window.location.hash === "#spmm" ? "spmm" : "spmv");
function syncOperator() {
  selectedOperator.value = window.location.hash === "#spmm" ? "spmm" : "spmv";
}
onMounted(() => window.addEventListener("hashchange", syncOperator));
onBeforeUnmount(() => window.removeEventListener("hashchange", syncOperator));
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

@media (max-width: 750px) {
  .qiwu-brand-bar {
    padding: 10px 10%;
  }

  .qiwu-logo {
    height: 40px;
  }
}
</style>
