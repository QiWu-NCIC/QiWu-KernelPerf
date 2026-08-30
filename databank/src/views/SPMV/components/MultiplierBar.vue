<template>
  <div class="multiplier-bar">
    <div class="bar-container">
      <div class="bar bar-100x" :style="{ width: bar100xWidth }"></div>
      <div class="bar bar-10x" :style="{ width: bar10xWidth }"></div>
      <div class="bar bar-1x" :style="{ width: bar1xWidth }"></div>
    </div>
    <div class="value-display">x {{ displayValue }}</div>
  </div>
</template>

<script setup>
import { computed } from "vue";

const props = defineProps({
  value: {
    type: Number,
    required: true,
    default: 0,
  },
  maxValue: {
    type: Number,
    default: 1000,
  },
});

// 计算每个倍数的宽度百分比（基于 maxValue）
const bar1xWidth = computed(() => {
  const percentage = Math.min((props.value / props.maxValue) * 100, 100);
  return `${percentage}%`;
});

const bar10xWidth = computed(() => {
  const percentage = Math.min(((props.value * 10) / props.maxValue) * 100, 100);
  return `${percentage}%`;
});

const bar100xWidth = computed(() => {
  const percentage = Math.min(
    ((props.value * 100) / props.maxValue) * 100,
    100,
  );
  return `${percentage}%`;
});

const displayValue = computed(() => {
  return Math.round(props.value * 100) / 100;
});
</script>

<style lang="scss" scoped>
.multiplier-bar {
  width: 100%;
  margin: 10px 0;
  position: relative;
}

.bar-container {
  position: relative;
  width: 100%;
  height: 24px;
  background-color: #e9e9e9;
  overflow: hidden;
}

.bar {
  position: absolute;
  left: 0;
  top: 0;
  height: 100%;
  transition: width 0.3s ease;
}

.bar-1x {
  background: #935400;
  z-index: 3;
}

.bar-10x {
  background: #b8741a;
  z-index: 2;
}

.bar-100x {
  background: #facd91;
  z-index: 1;
}

.value-display {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  text-align: right;
  font-size: 14px;
  color: #333333;
}
</style>
