<template>
  <nav class="operator-tabs" aria-label="Operator">
    <span>Operator</span>
    <a href="#spmv" :aria-current="current === 'spmv' ? 'page' : undefined">SpMV</a>
    <a href="#spmm" :aria-current="current === 'spmm' ? 'page' : undefined">SpMM</a>
  </nav>
</template>

<script setup>
defineProps({
  current: { type: String, required: true },
});
</script>

<style scoped>
.operator-tabs {
  display: flex;
  align-items: center;
  gap: 8px;
  max-width: 1440px;
  margin: 0 auto 18px;
  color: #526276;
  font-size: 12px;
}

.operator-tabs > span {
  margin-right: 4px;
  font-weight: 600;
}

.operator-tabs a {
  display: inline-flex;
  min-height: 34px;
  align-items: center;
  padding: 6px 13px;
  border: 1px solid #b9c7d6;
  border-radius: 4px;
  background: #ffffff;
  color: #254d9e;
  font-size: 12px;
  font-weight: 600;
  text-decoration: none;
}

.operator-tabs a:hover {
  border-color: #254d9e;
  background: #f2f6fd;
}

.operator-tabs a[aria-current="page"] {
  border-color: #254d9e;
  background: #254d9e;
  color: #ffffff;
}

.operator-tabs a:focus-visible {
  outline: 2px solid #256b87;
  outline-offset: 2px;
}
</style>
