<template>
  <div class="layout-wrapper">
    <div class="layout-header">
      <!-- logo -->
      <img class="logo" src="@/assets/images/logo.png" alt="logo" />
      <!-- 语言切换 -->
      <div class="lang">
        <div
          class="lang-item"
          v-for="(item, index) in options"
          :key="index"
          :class="{ active: item.value === activeName }"
          @click="handLang(item)"
        >
          {{ item.label }}
        </div>
      </div>
    </div>
    <div class="layout-main">
      <router-view v-slot="{ Component }">
        <keep-alive>
          <component :is="Component" />
        </keep-alive>
      </router-view>
      <el-backtop target=".layout-main" :right="100" :bottom="50" />
    </div>
  </div>
</template>
<script setup>
const { locale } = useI18n();

let options = [
  {
    label: "EN",
    value: "en",
  },
  {
    label: "中文",
    value: "zh",
  },
];

// 语言
let activeName = computed({
  get: () => {
    return locale.value;
  },
  set: (val) => {
    locale.value = val;
    localStorage.setItem("lang", locale.value);
  },
});

// 语言切换
function handLang(row) {
  if (row.value === activeName.value) return;
  activeName.value = row.value;
}
</script>
<style lang="scss" scoped>
.layout-wrapper {
  height: 100%;
  display: flex;
  flex-direction: column;
}
.layout-header {
  width: 100%;
  background: #254d9e;
  padding: 20px 10%;
  display: flex;
  justify-content: space-between;
  align-items: center;
  .logo {
    height: 50px;
  }
  .lang {
    border-radius: 2px;
    border: 1px solid #ffffff;
    display: flex;
    align-items: center;

    .lang-item {
      font-size: 16px;
      width: 40px;
      height: 21px;
      line-height: 21px;
      text-align: center;
      color: #fff;
      cursor: pointer;
      &.active {
        background: #fff;
        color: #254d9e;
      }
      &:first-child {
        border-radius: 2px 0 0 2px;
      }
      &:last-child {
        border-radius: 0 2px 2px 0;
      }
    }
  }
}

.layout-main {
  flex: 1;
  overflow: auto;
}

@media only screen and (max-width: 750px) {
  .layout-header {
    padding: 10px 10%;
  }
  .layout-header .logo {
    height: 40px;
  }
}
</style>
