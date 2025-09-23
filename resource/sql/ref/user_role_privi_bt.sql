CREATE DATABASE  IF NOT EXISTS `mes_omi` /*!40100 DEFAULT CHARACTER SET utf8 */;
USE `mes_omi`;
-- MySQL dump 10.13  Distrib 5.7.12, for Win32 (AMD64)
--
-- Host: 127.0.0.1    Database: mes_omi
-- ------------------------------------------------------
-- Server version	5.5.5-10.2.7-MariaDB

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

--
-- Table structure for table `user_role_privi_bt`
--

DROP TABLE IF EXISTS `user_role_privi_bt`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `user_role_privi_bt` (
  `role_id` varchar(16) NOT NULL,
  `privi_id` varchar(16) NOT NULL,
  `rec_user` varchar(16) DEFAULT NULL,
  `rec_time` varchar(19) DEFAULT NULL,
  `node_id` varchar(16) DEFAULT NULL,
  `node_time` timestamp(6) NULL DEFAULT current_timestamp(6) ON UPDATE current_timestamp(6),
  PRIMARY KEY (`role_id`,`privi_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8 AVG_ROW_LENGTH=1260 ROW_FORMAT=DYNAMIC;
/*!40101 SET character_set_client = @saved_cs_client */;

--
-- Dumping data for table `user_role_privi_bt`
--

LOCK TABLES `user_role_privi_bt` WRITE;
/*!40000 ALTER TABLE `user_role_privi_bt` DISABLE KEYS */;
INSERT INTO `user_role_privi_bt` VALUES ('mold_manager','mold_create',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_manager','mold_query',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_manager','mold_recv',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_manager','mold_repair',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_manager','mold_repair_rtn',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_manager','mold_return',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_manager','mold_scrap',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_manager','mold_shelf',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_manager','mold_shift',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_manager','mold_user',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_op','mold_create',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_op','mold_query',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_op','mold_recv',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_op','mold_repair',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_op','mold_repair_rtn',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_op','mold_returnww',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_op','mold_scrap',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_op','mold_shelf',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_op','mold_shift',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_op','mold_user',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_sys','mold_create',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_sys','mold_query',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_sys','mold_recv',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_sys','mold_repair',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_sys','mold_repair_rtn',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_sys','mold_return',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_sys','mold_scrap',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_sys','mold_shelf',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_sys','mold_shift',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('mold_sys','mold_user',NULL,NULL,NULL,'2020-02-19 06:01:12.494828'),('operator','rpt_ker',NULL,NULL,NULL,'2020-02-16 01:23:48.420224'),('operator','rpt_ksr',NULL,NULL,NULL,'2020-02-16 01:23:48.420224'),('operator','rpt_omi',NULL,NULL,NULL,'2020-02-16 01:23:48.420224'),('planner','rpt_ksr',NULL,NULL,NULL,'2020-02-16 01:23:48.420224'),('planner','rpt_omi',NULL,NULL,NULL,'2020-02-16 01:23:48.420224'),('planner','rpt_shr',NULL,NULL,NULL,'2020-02-16 01:23:48.420224'),('planner','rpt_spc',NULL,NULL,NULL,'2020-02-16 01:23:48.420224'),('qc','rpt_spc',NULL,NULL,NULL,'2020-02-16 01:23:48.420224'),('supervisor','rpt_ker',NULL,NULL,NULL,'2020-02-16 01:23:48.420224'),('supervisor','rpt_ksr',NULL,NULL,NULL,'2020-02-16 01:23:48.420224'),('supervisor','rpt_omi',NULL,NULL,NULL,'2020-02-16 01:23:48.420224'),('supervisor','rpt_shr',NULL,NULL,NULL,'2020-02-16 01:23:48.420224'),('supervisor','rpt_spc',NULL,NULL,NULL,'2020-02-16 01:23:48.420224');
/*!40000 ALTER TABLE `user_role_privi_bt` ENABLE KEYS */;
UNLOCK TABLES;
/*!40103 SET TIME_ZONE=@OLD_TIME_ZONE */;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;

-- Dump completed on 2020-05-12  8:16:24
