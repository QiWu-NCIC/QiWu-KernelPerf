import {
  createRouter,
  createWebHistory,
  createWebHashHistory,
} from "vue-router";
import Layout from "@/layout/index.vue";

const routes = [
  {
    path: "/",
    component: Layout,
    redirect: "",
    children: [
      {
        path: "",
        name: "Index",
        component: () => import("@/views/index/index.vue"),
        meta: {
          title: "首页",
        },
      },
      {
        path: "/spmv",
        name: "SPMV",
        component: () => import("@/views/SPMV/index.vue"),
        meta: {
          title: "SPMV",
        },
      },
    ],
  },
];

const router = createRouter({
  // history: createWebHistory(),
  history: createWebHashHistory(),
  routes,
  scrollBehavior(to, from, savedPosition) {
    return { top: 0 };
  },
});

export default router;
